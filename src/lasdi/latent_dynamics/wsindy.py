import numpy as np
import torch
from scipy.integrate import odeint
from . import LatentDynamics
from ..inputs import InputParser
from ..fd import FDdict

'''
We use the same notation here as in the manuscript "Weak-Form Latent Space Dynamics Identification"
https://arxiv.org/abs/2311.12880

Pytorch version of weak-form SINDy originally implemented by Margaret Trautner. Adapted here
for refactored version of GPLaSDI
'''


class wSINDy(LatentDynamics):
    fd_type = ''
    fd = None
    fd_oper = None

    def __init__(self, dim, nt, dt, config):
        super().__init__(dim, nt)

        #TODO(kevin): generalize for high-order dynamics
        self.ncoefs = self.dim * (self.dim + 1)

        assert('wsindy' in config)
        parser = InputParser(config['wsindy'], name='wsindy_input')

        '''
            fd_type is the string that specifies finite-difference scheme for time derivative:
                - 'sbp12': summation-by-parts 1st/2nd (boundary/interior) order operator
                - 'sbp24': summation-by-parts 2nd/4th order operator
                - 'sbp36': summation-by-parts 3rd/6th order operator
                - 'sbp48': summation-by-parts 4th/8th order operator
        '''
        
        self.dt = dt
        self.fd_type = parser.getInput(['fd_type'], fallback='sbp12')
        self.fd = FDdict[self.fd_type]
        self.fd_oper, _, _ = self.fd.getOperators(self.nt)

        # NOTE(kevin): by default, this will be L1 norm.
        self.coef_norm_order = parser.getInput(['coef_norm_order'], fallback=1)

        # TODO(kevin): other loss functions
        self.MSE = torch.nn.MSELoss()

        self.test_func = parser.getInput(['test_func'], fallback='PC-poly')
        self.test_func_width = parser.getInput(['test_func_width'], fallback=50/(self.nt - 1))
        self.overlap = parser.getInput(['overlap'], fallback=0.8)
        self.pq = parser.getInput(['pq'], fallback=5)
        self.LS_loss_type = parser.getInput(['LS_loss_type'], fallback='weak') #weak or strong
        
        self.T = self.dt * (self.nt - 1)
        self.Phis, self.dPhis = self.get_test_functions(self.T, self.nt, self.test_func_width,self.overlap, self.pq, test_func = self.test_func)


        return
    
    def calibrate(self, Z, dt, compute_loss=True, numpy=False):
        ''' loop over all train cases, if Z dimension is 3 '''
        if (Z.dim() == 3):
            n_train = Z.size(0)

            if (numpy):
                coefs = np.zeros([n_train, self.ncoefs])
            else:
                coefs = torch.zeros([n_train, self.ncoefs])
            loss_wsindy, loss_coef = 0.0, 0.0

            for i in range(n_train):
                result = self.calibrate(Z[i], dt, compute_loss, numpy)
                if (compute_loss):
                    coefs[i] = result[0]
                    loss_wsindy += result[1]
                    loss_coef += result[2]
                else:
                    coefs[i] = result
            
            if (compute_loss):
                return coefs, loss_wsindy, loss_coef
            else:
                return coefs

        ''' evaluate for one train case '''
        assert(Z.dim() == 2)
        dZdt = self.compute_time_derivative(Z, dt)
        time_dim, space_dim = dZdt.shape

        Z_i = torch.cat([torch.ones(time_dim, 1), Z], dim = 1)
       
        Gk, bk = self.compute_Gk_bk(self.dim,self.Phis,self.dPhis,Z_i,Z) # (n_z, n_z, H, J) permuted to (n_z, H, n_z, J) and reshaped to (n_z*H, n_z*J)
        coefs = torch.linalg.lstsq(Gk,bk).solution
        coefs = coefs.reshape((self.dim, Z_i.shape[-1])).T


        if (compute_loss):
            if self.LS_loss_type == 'strong':
                loss_wsindy = self.MSE(dZdt, Z_i @ coefs)
            else:
                loss_wsindy = self.MSE(-self.dPhis @ Z, self.Phis @ Z_i @ coefs)
            # NOTE(kevin): by default, this will be L1 norm.
            loss_coef = torch.norm(coefs, self.coef_norm_order)

        # output of lstsq is not contiguous in memory.
        coefs = coefs.detach().flatten()
        if (numpy):
            coefs = coefs.numpy()

        if (compute_loss):
            return coefs, loss_wsindy, loss_coef
        else:
            return coefs

    def compute_time_derivative(self, Z, Dt):

        '''

        Builds the SINDy dataset, assuming only linear terms in the SINDy dataset. The time derivatives are computed through
        finite difference.

        Z is the encoder output (2D tensor), with shape [time_dim, space_dim]
        Dt is the size of timestep (assumed to be a uniform scalar)

        The output dZdt is a 2D tensor with the same shape of Z.

        '''
        return 1. / Dt * torch.sparse.mm(self.fd_oper, Z)

    def simulate(self, coefs, z0, t_grid):

        '''

        Integrates each system of ODEs corresponding to each training points, given the initial condition Z0 = encoder(U0)

        '''
        # copy is inevitable for numpy==1.26. removed copy=False temporarily.
        c_i = coefs.reshape([self.dim+1, self.dim]).T
        dzdt = lambda z, t : c_i[:, 1:] @ z + c_i[:, 0]

        Z_i = odeint(dzdt, z0, t_grid)

        return Z_i
    
    def export(self):
        param_dict = super().export()
        param_dict['fd_type'] = self.fd_type
        param_dict['coef_norm_order'] = self.coef_norm_order
        param_dict['test_func'] = self.test_func
        param_dict['test_func_width'] = self.test_func_width
        param_dict['overlap'] = self.overlap
        param_dict['pq'] = self.pq
        param_dict['Phis'] = self.Phis
        param_dict['dPhis'] = self.dPhis
        return param_dict
    
    
    def getUniformGrid(self,T, L, s, p):

        '''
        generates uniform grid for test functions
        s is overlap
        L is test function width
        '''
        

        overlap = s # int(np.floor(L*(1 - np.sqrt(1 - s**(1/p)))))
        #print("support and overlap", L, overlap)
        # create grid
        grid = []
        a = 0
        b = L
        grid.append([a, b])
        while b - overlap + L <= T:
            a = b - overlap
            b = a + L
            grid.append([a, b])

        grid = np.asarray(grid)
        
        a_s = grid[:,0]
        b_s = grid[:,1]

        return a_s, b_s


    
    def get_test_functions(self, T,n_t, test_func_width, overlap,pq, test_func = 'PC-poly',H = 30): 
        '''
        H: number of test functions compactly supported on time 
        interval [0,T], assumes equally spaces
        n_t: number of time points, assumes equally spaced

        returns: Phis: dim (H, n_t), each of H test functions evaluated at n_t time points
                dPhis: dim (H, n_t), each of H test function time derivatives evaluated at n_t time points
        '''
        
        t = torch.linspace(0,T,n_t)

        if test_func == 'bump':
            L = test_func_width #T/50 # length of test function support
            s = test_func_width*overlap # overlap
            a_s, b_s = self.getUniformGrid(T, L, s, 1)

            t = torch.linspace(0,T,n_t)
            H = len(a_s)
            print("Number of Bump test functions:", H)
            
            Phis = torch.zeros((H,n_t))
            dPhis = torch.zeros((H,n_t))
            d2Phis = torch.zeros((H,n_t))
            # eta = 1
            eta = 5
            a = L/2
            const = eta #bumps are of form e^( -eta/(1 - (x/a)^2 ) + const)
            # Make function integrate to 1
            # Numerical integration 
            nugget = 1e-7
            a_space = np.linspace(-a+nugget,a-nugget,1000)
            bump = np.exp(-eta/(1-(a_space/a)**2))
            C = 1/np.trapz(bump,a_space)/np.exp(const)

            h = torch.linspace(a,T-a,H)

            for j, ji in enumerate(h):
                for i, ti in enumerate(t):
                    x = (ti-ji)/a
                    denom = 1 - x**2
                    f = -eta/denom + const 
                    fp = -eta/(denom**2)*2*x/a
                    fpp = ( -eta/(denom**2)*2/a/a ) + ( -eta/(denom**3)*2*x/a * -2*x/a*-2 )
                    if denom > 0:
                        Phis[j,i] = C*torch.exp(f)
                        dPhis[j,i] = C*torch.exp(f)*fp
                        d2Phis[j,i] = C*( torch.exp(f)*(fp**2) + torch.exp(f)*fpp )
                    else:
                        Phis[j,i] = 0
                        dPhis[j,i] = 0
                        d2Phis[j,i] = 0

        if test_func == 'PC-poly':
            L = test_func_width #T/50 # length of test function support
            s = test_func_width*overlap # overlap
            a_s, b_s = self.getUniformGrid(T, L, s, 1)

            t = torch.linspace(0,T,n_t)
            H = len(a_s)
            print("Number of test functions:", H)
            Phis = torch.zeros((H,n_t))
            dPhis = torch.zeros((H,n_t))
            

            # parallize the time computations
            p, q = pq, pq 
            for h in range(H):
                a = a_s[h]
                b = b_s[h]
                C = 1/(p**p*q**q)*((p+q)/(b-a))**(p+q)
                Phis[h,:] = C* (t-a)**p*(b-t)**q*(t>=a)*(t<=b)
                dPhis[h,:] = C* (p*(t-a)**(p-1)*(b-t)**q - q*(t-a)**p*(b-t)**(q-1))*(t>=a)*(t<=b)

        return Phis, dPhis

   
    def compute_Gk_bk(self, n_s,Phi,dPhi,Theta,U):
        '''
        n_s: reduced dim
        Phi: dimension (H,n_T)
        dPhi: dimension (H,n_T)
        Theta: dimension (n_T, J)

        Gk = I_{n_s} \otimes \Phi \Theta :  dimension (H*n_s,J*n_s)
        bk = -vec(dPhi*U): dimension (H*n_s)
        '''
        n_s = U.shape[1]
        H = Phi.shape[0]
        J = Theta.shape[1]

        Ins = torch.eye(n_s)
        PhiTheta = Phi@Theta
        Gk = torch.tensordot(Ins,PhiTheta,dims = 0)
        Gk = Gk.permute(0,2,1,3).reshape((H*n_s,J*n_s))

        bk = dPhi @ U 
        bk = -bk.permute(1,0).reshape((H*n_s,1)) 

        return Gk, bk

   
