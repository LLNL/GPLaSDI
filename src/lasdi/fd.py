# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  numpy           as      np
import  scipy.sparse    as      sps
from    scipy.sparse    import  spmatrix;
import  torch



# -------------------------------------------------------------------------------------------------
# Stencil class
# -------------------------------------------------------------------------------------------------

class Stencil:
    # Class variables
    leftBdrDepth        : int               = 0         # Number of time steps (at the start/stop of the time series) we use a special stencil on.
    leftBdrWidth        : list[int]         = []        # i'th element specifies the number of terms in the stencil for the i'th time step
    leftBdrStencils     : list[list[float]] = [[]]      # i'th element is a list wit ki = len(leftBdrWidth[i]) specifying the stencil weights of the first ki time series terms in the stencil for x(t_i)
    leftBdrNorm         : list[float]       = []        # ??? TODO: what is this?

    """
    Suppose that InteriorStencils is an array of length Ns and interiorIndexes is a list of 
    length Ns. Further suppose the underlying time series contains Nx points, x(t_0), ... , 
    x(t_{Nx -1}). Assuming index i is an "interior index" (not too close to 0 or Nx - 1), then we 
    approximate the time derivative of z at time t_i as follows:
        z'(t_i) \approx c_0 z(t_{i + i(0)}) + ... + c_{Ns - 1} z(t_{i + i(Ns - 1)})
    where c_k = interiorStencils[k] and i(k) = interiorIndexes[k]. Note that the indices may be 
    negative or positive. Thus, interiorStencils and interiorIndexes tell us how to construct the 
    finite difference scheme away from the boundary. 
    x
    For instance, the central difference scheme corresponds to interiorStencils = [-1/2, 1/2], 
    interiorIndexes = [-1, 1] and 
        z'(t_i) \approx (1/2)(-z(t_{i - 1}) + z_{t_{i + 1}}) 

    Note: We assume that interiorIndexes is in ASCENDING order. 
    """
    interiorStencils    : np.ndarray    = np.array([])  # Specify the weights of the terms in the stencil
    interiorIndexes     : list[int]     = []            # Specify the relative indices of the terms in the stencil. Must be in ascending order.


    
    def getOperators(self, Nx : int, periodic : bool = False) -> tuple[spmatrix, torch.Tensor, torch.Tensor]:
        """
        The stencil class acts as an abstract base class for finite difference schemes. We assume
        that the user has a time series, x(t_0), ... , x(t_{Nx - 1}) \in \mathbb{R}^d. We will 
        further assume that the time stamps are evenly spaced, t_0, ... , t_{Nx - 1}. That is, 
        there is some \Delta t such that t_k = t_0 + \Delta_t*k for each k. We also assume the user 
        wants to approximate the time derivative of x at t_0, ... , t_{Nx - 1} using a finite 
        difference scheme.

        The getOperators method builds an a sparse Tensor housing the "operator matrix". This is 
        an Nx x Nx matrix that we use to apply the finite difference scheme to one component of 
        the time series. How does this work? Suppose the time series is x(t_0), ... , 
        x(t_{Nx - 1}) \in \mathbb{R}^d. Then, for each i \in \{ 1, 2, ... , d}, we construct Dxi 
        such that Dxi * xi is the vector whose i'th entry holds the approximation (using the 
        selected finite difference scheme) of x_i'(t_i). Here, xi = [x_i(t_0), ... , 
        x_i(t_{Nx - 1})]. To build Dxi, we first set up a matrix to give the correct approximation
        to x_i'(t_j) whenever j is an interior index. The rest of this function adjusts the 
        first/last few rows of Dxi so that it also gives the correct approximation at the 
        boundaries (j close to to 0 or Nx - 1).

        The Stencil class is not a standalone class; you shouldn't use it directly. Rather, you 
        should use one of the sub-classes defined below. Each one implements a specific finite 
        difference scheme to approximate the time derivatives at t_0, ... , t_{Nx - 1} (they may 
        use different rules on the left and right boundary). Each sub-class should set the 
        interiorStencils and interiorIndexes attributes.
        
        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Nx: An integer specifying the number of points in the time series we want to apply a 
        finite difference scheme to.

        periodic: A boolean specifying if we should treat the time series as periodic or not.    

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Three elements. The first Dxi, the "Operator Matrix" described above. The second holds the 
        "norm" tensor and the third holds the "PeriodicOffset" tensor.
        TODO: what are the last two of these used for? And what are they?
        """

        norm            = np.ones(Nx,)
        periodicOffset  = np.zeros(Nx,)
        
        # Start building the "Operator Matrix" (See docstring)
        Dxi : spmatrix  = sps.diags(self.interiorStencils,
                                    self.interiorIndexes,
                                    shape = (Nx, Nx), 
                                    format = 'lil')
        
        """
        If we assume the time series is periodic, then we need to adjust the first/last few 
        rows to "wrap around". In this case, every index is an interior index. However, this means
        that we will need x(t_{-k}) when approximating x'(t_0), for instance, or x(t_{Nx + k}) 
        when approximating x'(t_{Nx - 1}). Periodicity means we assume x(t_{-k}) = x(t_{Nx - k}) 
        and x(t_{Nx + k}) = x(t_k). Thus, we can use the last few components of x in place of terms
        like x(t_{-k}). For 
        instance, if we wanted to use a central difference scheme, then we have 
            x'(t_0) \approx (1/2)(-x(t_{-1}) + x(t_{1})) = (1/2)(-x(t_{Nx - 1}) + x(t_1))
        In other words, by modifying the first/last few rows of Dxi, we can build an operator
        matrix that can recover our approximation of x' near the boundaries (think about it).
        """
        if (periodic):
            """
            Get the indices less than zero and greater than Nx that we will need to update the
            first/last few rows ofDxi. Remember that the elements of interiorIndexes are in 
            ascending order. Thus, if interiorIndexes[0] = -k, then we need to update the first 
            k rows of Dxi to account for the periodic BCs (think about it). Likewise, if 
            interiorIndexes[-1] = j, then we also need to update the final j rows of Dxi (think 
            about it).

            If the smallest and largest elements of interiorIndexes are -k and j, respectively, 
            then the line below produces the list [0, 1, ... , k - 1, -j, -j + 1, ... , -1].
            """
            bdrIdxes : list[int] = ([k for k in range(-self.interiorIndexes[0])] +
                                    [k for k in range(-self.interiorIndexes[-1], 0)])
            
            # Cycle through the rows of Dxi that we need to correct.
            for idx in bdrIdxes:
                # Determine which columns of Dxi we need to access to apply the stencil for the 
                # idx'th row of Dxi (corresponding to time t_i). 
                colIdx : list[int]  = [k + idx for k in self.interiorIndexes]

                # Now set the corresponding elements with appropriate wrap around (note that if 
                # one index is negative, then numpy's indexing rules will automatically wrap it
                # since x[-k] = x[x.len - k]).
                Dxi[idx, colIdx]    = self.interiorStencils
                
                # Set up the periodicOffset. If idx is negative, we are fixing one of the last
                # few rows of Dxi. In this case, periodicOffset holds the sum of the weights 
                # of the indices at and to the right of the current index in the interiorStencil.
                # Otherwise, we set it to the sum of the weights of the indices to the left.
                if (idx < 0):
                    temp = [k >= 0 for k in colIdx]
                    periodicOffset[idx] = np.sum(self.interiorStencils[temp])
                else:
                    temp = [k < 0 for k in colIdx]
                    periodicOffset[idx] = -np.sum(self.interiorStencils[temp])
        
        # Otherwise, we need use special stencils for the left and right boundary terms.
        else:
            # If leftBrdDepth = k, then we use a special stencil to approximate x'(t_0), ... , 
            # x'(t_{k - 1}) and for x'(t_{Nx - k}), ... , x'(t_{Nx - 1}). We begin by zeroing out
            # the first/last k rows of Dxi.
            Dxi[:self.leftBdrDepth,:]   = 0.
            Dxi[-self.leftBdrDepth:,:]  = 0.

            # Now, populate the first/last few rows of Dxi to implement the boundary stencil. 
            for depth in range(self.leftBdrDepth):
                # leftBdrWidth[depth] = k means that the stencil for x(t_{depth}) depends on 
                # x(t_0), ... , x(t_{k - 1}).
                width   : int  = self.leftBdrWidth[depth]
                
                # Update the first width elements of the depth'th row of Dxi using the depth'th 
                # elements of leftBdrStencils (which holds the stencil weights for this row's 
                # stencil)
                Dxi[depth,:width] = self.leftBdrStencils[depth]

                # Update the (depth - 1)'th to last row of Dxi using the reversed stencil (which 
                # gives us an equivalent stencil for right boundaries).
                # NOTE: Currently only consider skew-symmetric operators.
                Dxi[-1-depth,-width:] = -Dxi[depth,(width-1)::-1]
            
            # TODO: what is going on here?
            norm[:self.leftBdrDepth]    = self.leftBdrNorm
            norm[-self.leftBdrDepth:]   = norm[(self.leftBdrDepth-1)::-1]

        # Convert Dxi to a tensor. 
        Dxi : torch.Tensor = self.convert(sps.coo_matrix(Dxi))

        # All done!
        return Dxi, torch.Tensor(norm), torch.Tensor(periodicOffset)
    


    def convert(self, scipy_coo : spmatrix) -> torch.Tensor:
        """
        Converts scipy_coo, a sparse numpy array, to a sparse torch Tensor.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        scipy_coo: A sparse numpy array.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A tensor version of scipy_coo.
        """
        
        if (type(scipy_coo) is not sps._coo.coo_matrix):
            raise RuntimeError("The input sparse matrix must be in scipy COO format!")


        # Fetch the values of the non-zero entries of scipy_coo.  
        values : np.ndarray = scipy_coo.data

        # Get the row, column numbers of the non-zero entries of scipy_coo. vertically stack them 
        # into a 2 x N matrix (N being the number of non-zero entries in scipy_coo) whose j'th 
        # column holds the row and column number of the j'th non-zero cell in scipy_coo.
        indices : np.ndarray = np.vstack((scipy_coo.row, scipy_coo.col))

        # Convert indices to a tensor
        i : torch.Tensor = torch.LongTensor(indices)

        # Convert the list of non-zero values in scipy_coo to a tensor.
        v : torch.Tensor = torch.FloatTensor(values)

        # Get the shape of scipy.coo
        shape = scipy_coo.shape

        # Convert scipy_coo to a tensor and return!
        return torch.sparse_coo_tensor(i, v, torch.Size(shape))



# -------------------------------------------------------------------------------------------------
# Stencil sub-classes
# -------------------------------------------------------------------------------------------------

class SBP12(Stencil):
    def __init__(self):
        self.leftBdrDepth = 1
        self.leftBdrWidth = [2]
        self.leftBdrStencils = [[-1., 1.]]
        self.leftBdrNorm = [0.5]

        self.interiorStencils = np.array([-0.5, 0.5])
        self.interiorIndexes = [-1, 1]
        return


   
class SBP24(Stencil):
    def __init__(self):
        self.leftBdrDepth = 4
        self.leftBdrWidth = [4, 3, 5, 6]
        self.leftBdrStencils = [[-24./17., 59./34., -4./17., -3./34.],
                                [-1./2., 0., 1./2.],
                                [4./43., -59./86., 0., 59./86., -4./43.],
                                [3./98., 0., -59./98., 0., 32./49., -4./49.]]
        self.leftBdrNorm = [17./48., 59./48., 43./48., 49./48.]

        self.interiorStencils = np.array([1./12., -2./3., 2./3., -1./12.])
        self.interiorIndexes = [-2, -1, 1, 2]
        return



class SBP36(Stencil):
    def __init__(self):
        self.leftBdrDepth = 6
        self.leftBdrWidth = [6, 6, 6, 7, 8, 9]
        self.leftBdrStencils = [[-21600./13649., 104009./54596., 30443./81894., -33311./27298., 16863./27298., -15025./163788.],
                                [-104009./240260., 0., -311./72078., 20229./24026., -24337./48052., 36661./360390.],
                                [-30443./162660., 311./32532., 0., -11155./16266., 41287./32532., -21999./54220.],
                                [33311./107180., -20229./21436., 485./1398., 0., 4147./21436., 25427./321540., 72./ 5359.],
                                [-16863./78770., 24337./31508., -41287./47262., -4147./15754., 0., 342523./472620., -1296./ 7877., 144./ 7877.],
                                [15025./525612., -36661./262806., 21999./87602., -25427./262806., -342523./525612., 0., 32400./ 43801., -6480./ 43801., 720./ 43801.]]
        self.leftBdrNorm = [13649./43200., 12013./8640., 2711./4320.,
                            5359./4320., 7877./8640., 43801./43200.]

        self.interiorStencils = np.array([-1./60., 3./20., -3./4., 3./4., -3./20., 1./60.])
        self.interiorIndexes = [-3, -2, -1, 1, 2, 3]
        return



class SBP48(Stencil):
    def __init__(self):
        self.leftBdrDepth = 8
        self.leftBdrWidth = [8, 8, 8, 8, 9, 10, 11, 12]
        self.leftBdrNorm = [1498139./5080320., 1107307./725760., 20761./80640., 1304999./ 725760.,
                             299527./725760., 103097./80640., 670091./725760., 5127739./5080320.]

        x1 = 541. / 1000.
        x2 = -27. /  400.
        x3 = 187. /  250.

        self.leftBdrStencils = [[None] * self.leftBdrWidth[k] for k in range(self.leftBdrDepth)]
        self.leftBdrStencils[0][0] = -2540160. / 1498139.
        self.leftBdrStencils[0][1] = 9. * (2257920. * x1 + 11289600. * x2
                                          + 22579200. * x3 - 15849163.) / 5992556.
        self.leftBdrStencils[0][2] = 3. * (-33868800. * x1 - 162570240. * x2
                                          - 304819200. * x3 + 235236677.) / 5992556.
        self.leftBdrStencils[0][3] = (609638400. * x1 + 2743372800. * x2
                                      + 4572288000. * x3 - 3577778591.) / 17977668.
        self.leftBdrStencils[0][4] = 3. * (-16934400 * x1 - 67737600. * x2
                                          - 84672000. * x3 + 67906303.) / 1498139.
        self.leftBdrStencils[0][5] = 105. * (967680. * x1 + 2903040. * x2 - 305821.) / 5992556.
        self.leftBdrStencils[0][6] = 49. * (-1244160. * x1 + 18662400. * x3 - 13322233.) / 17977668.
        self.leftBdrStencils[0][7] = 3. * (-6773760. * x2 - 33868800. * x3 + 24839327.) / 5992556.

        self.leftBdrStencils[1][0] = 9. * (-2257920. * x1 - 11289600. * x2
                                          - 22579200. * x3 + 15849163.) / 31004596.
        self.leftBdrStencils[1][1] = 0.
        self.leftBdrStencils[1][2] = 3. * (7257600. * x1 + 33868800. * x2
                                          + 60963840. * x3 - 47167457.) / 2214614.
        self.leftBdrStencils[1][3] = 3. * (-9676800. * x1 - 42336000. * x2
                                          - 67737600. * x3 + 53224573.) / 1107307.
        self.leftBdrStencils[1][4] = 7. * (55987200. * x1 + 217728000. * x2
                                          + 261273600. * x3 - 211102099.) / 13287684.
        self.leftBdrStencils[1][5] = 3. * (-11612160. * x1 - 33868800. * x2 + 3884117.) / 2214614.
        self.leftBdrStencils[1][6] = 150. * (24192. * x1 - 338688. * x3 + 240463.) / 1107307.
        self.leftBdrStencils[1][7] = (152409600. * x2 + 731566080. * x3 - 536324953.) / 46506894.

        self.leftBdrStencils[2][0] = (33868800. * x1 + 162570240. * x2
                                      + 304819200. * x3 - 235236677.) / 1743924.
        self.leftBdrStencils[2][1] = (-7257600. * x1 - 33868800. * x2
                                      - 60963840. * x3 + 47167457.) / 124566.
        self.leftBdrStencils[2][2] = 0.
        self.leftBdrStencils[2][3] = (24192000. * x1 + 101606400. * x2
                                      + 152409600. * x3 - 120219461.) / 124566.
        self.leftBdrStencils[2][4] = (-72576000. * x1 - 270950400. * x2
                                      - 304819200. * x3 + 249289259.) / 249132.
        self.leftBdrStencils[2][5] = 9. * (806400. * x1 + 2257920. * x2 - 290167.) / 41522.
        self.leftBdrStencils[2][6] = 6. * (-134400. * x1 + 1693440. * x3 - 1191611.) / 20761.
        self.leftBdrStencils[2][7] = 5. * (-2257920. * x2 - 10160640. * x3 + 7439833.) / 290654.

        self.leftBdrStencils[3][0] = (-609638400. * x1 - 2743372800. * x2
                                      - 4572288000. * x3 + 3577778591.) / 109619916.
        self.leftBdrStencils[3][1] = 3. * (9676800. * x1 + 42336000. * x2
                                          + 67737600. * x3 - 53224573.) / 1304999.
        self.leftBdrStencils[3][2] = 3. * (-24192000. * x1 - 101606400. * x2
                                          - 152409600. * x3 + 120219461.) / 2609998.
        self.leftBdrStencils[3][3] = 0.
        self.leftBdrStencils[3][4] = 9. * (16128000. * x1 + 56448000. * x2
                                          + 56448000. * x3 - 47206049.) / 5219996.
        self.leftBdrStencils[3][5] = 3. * (-19353600. * x1 - 50803200. * x2 + 7628371.) / 2609998.
        self.leftBdrStencils[3][6] = 2. * (10886400. * x1 - 114307200. * x3 + 79048289.) / 3914997.
        self.leftBdrStencils[3][7] = 75. * (1354752. * x2 + 5419008. * x3 - 3952831.) / 18269986.

        self.leftBdrStencils[4][0] = 3. * (16934400. * x1 + 67737600. * x2
                                          + 84672000. * x3 - 67906303.) / 2096689.
        self.leftBdrStencils[4][1] = 7. * (-55987200. * x1 - 217728000. * x2
                                          - 261273600. * x3 + 211102099.) / 3594324.
        self.leftBdrStencils[4][2] = 3. * (72576000. * x1 + 270950400. * x2
                                          + 304819200. * x3 - 249289259.) / 1198108.
        self.leftBdrStencils[4][3] = 9. * (-16128000. * x1 - 56448000. * x2
                                          - 56448000. * x3 + 47206049.) / 1198108.
        self.leftBdrStencils[4][4] = 0.
        self.leftBdrStencils[4][5] = 105. * (414720. * x1 + 967680. * x2 - 165527.) / 1198108.
        self.leftBdrStencils[4][6] = 15. * (-967680. * x1 + 6773760. * x3 - 4472029.) / 1198108.
        self.leftBdrStencils[4][7] = (-304819200. * x2 - 914457600. * x3 + 657798011.) / 25160268.
        self.leftBdrStencils[4][8] = -2592. / 299527.

        self.leftBdrStencils[5][0]  = 5. * (-967680. * x1 - 2903040. * x2 + 305821.) / 1237164.
        self.leftBdrStencils[5][1]  = (11612160. * x1 + 33868800. * x2 - 3884117.) / 618582.
        self.leftBdrStencils[5][2]  = 9. * (-806400. * x1 - 2257920. * x2 + 290167.) / 206194.
        self.leftBdrStencils[5][3]  = (19353600. * x1 + 50803200. * x2 - 7628371.) / 618582.
        self.leftBdrStencils[5][4]  = 35. * (-414720. * x1 - 967680. * x2 + 165527.) / 1237164.
        self.leftBdrStencils[5][5]  = 0.
        self.leftBdrStencils[5][6]  = 80640. * x1 / 103097.
        self.leftBdrStencils[5][7]  = 80640. * x2 / 103097.
        self.leftBdrStencils[5][8]  = 3072. / 103097.
        self.leftBdrStencils[5][9] = -288. / 103097.

        self.leftBdrStencils[6][0]  = 7. * (1244160. * x1 - 18662400. * x3 + 13322233.) / 8041092.
        self.leftBdrStencils[6][1]  = 150. * (-24192. * x1 + 338688. * x3 - 240463.) / 670091.
        self.leftBdrStencils[6][2]  = 54. * (134400. * x1 - 1693440. * x3 + 1191611.) / 670091.
        self.leftBdrStencils[6][3]  = 2. * (-10886400. * x1 + 114307200. * x3 - 79048289.) / 2010273.
        self.leftBdrStencils[6][4]  = 15. * (967680. * x1 - 6773760. * x3 + 4472029.) / 2680364.
        self.leftBdrStencils[6][5]  = -725760. * x1 / 670091.
        self.leftBdrStencils[6][6]  = 0.
        self.leftBdrStencils[6][7]  = 725760. * x3 / 670091.
        self.leftBdrStencils[6][8]  = -145152. / 670091.
        self.leftBdrStencils[6][9] = 27648. / 670091.
        self.leftBdrStencils[6][10] = -2592. / 670091.

        self.leftBdrStencils[7][0]  = 3. * (6773760. * x2 + 33868800. * x3 - 24839327.) / 20510956.
        self.leftBdrStencils[7][1]  = (-152409600. * x2 - 731566080. * x3 + 536324953.) / 30766434.
        self.leftBdrStencils[7][2]  = 45. * (2257920. * x2 + 10160640. * x3 - 7439833.) / 10255478.
        self.leftBdrStencils[7][3]  = 75. * (-1354752. * x2 - 5419008. * x3 + 3952831.) / 10255478.
        self.leftBdrStencils[7][4]  = (304819200. * x2 + 914457600. * x3 - 657798011.) / 61532868.
        self.leftBdrStencils[7][5]  = -5080320. * x2 / 5127739.
        self.leftBdrStencils[7][6]  = -5080320. * x3 / 5127739.
        self.leftBdrStencils[7][7]  = 0.
        self.leftBdrStencils[7][8]  = 4064256. / 5127739.
        self.leftBdrStencils[7][9] = -1016064. / 5127739.
        self.leftBdrStencils[7][10] = 193536. / 5127739.
        self.leftBdrStencils[7][11] = -18144. / 5127739.

        self.interiorStencils = np.array([1./280., -4./105., 1./5., -4./5., 4./5., -1./5., 4./105., -1./280.])
        self.interiorIndexes = [-4, -3, -2, -1, 1, 2, 3, 4]
        return



FDdict = {'sbp12': SBP12(),
          'sbp24': SBP24(),
          'sbp36': SBP36(),
          'sbp48': SBP48()}