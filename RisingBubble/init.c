#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

double raiseto(double x, double a)
{
  return(exp(a*log(x)));
}

int main()
{
  double gamma    = 1.4;
  double R        = 287.058;
  double rho_ref  = 1.1612055171196529;
  double p_ref    = 100000.0;
  double grav_x   = 0.0;
  double grav_y   = 9.8;
  int    HB       = 0;
  
  int   NI,NJ,ndims;
  char  ip_file_type[50]; strcpy(ip_file_type,"ascii");
  FILE *in;

  printf("Reading file \"solver.inp\"...\n");
  in = fopen("solver.inp","r");
  if (!in) {
    printf("Error: Input file \"solver.inp\" not found.\n");
    return(0);
  } else {
    char word[500];
    fscanf(in,"%s",word);
    if (!strcmp(word, "begin")) {
      while (strcmp(word, "end")) {
        fscanf(in,"%s",word);
        if (!strcmp(word, "ndims")) fscanf(in,"%d",&ndims);
        else if (!strcmp(word, "size")) {
          fscanf(in,"%d",&NI);
          fscanf(in,"%d",&NJ);
        } else if (!strcmp(word, "ip_file_type")) fscanf(in,"%s",ip_file_type);
      }
    } else printf("Error: Illegal format in solver.inp. Crash and burn!\n");
  }
  fclose(in);

  printf("Reading file \"physics.inp\"...\n");
  in = fopen("physics.inp","r");
  if (!in) {
    printf("Error: Input file \"physics.inp\" not found.\n");
    return(0);
  } else {
    char word[500];
    fscanf(in,"%s",word);
    if (!strcmp(word, "begin")) {
      while (strcmp(word, "end")) {
        fscanf(in,"%s",word);
        if      (!strcmp(word, "rho_ref")) fscanf(in,"%lf",&rho_ref);
        else if (!strcmp(word, "p_ref"  )) fscanf(in,"%lf",&p_ref  );
        else if (!strcmp(word, "gamma"  )) fscanf(in,"%lf",&gamma  );
        else if (!strcmp(word, "R"      )) fscanf(in,"%lf",&R      );
        else if (!strcmp(word, "HB"     )) fscanf(in,"%d" ,&HB     );
        else if (!strcmp(word, "gravity")) {
          fscanf(in,"%lf",&grav_x );
          fscanf(in,"%lf",&grav_y );
        }
      }
    } else printf("Error: Illegal format in physics.inp. Crash and burn!\n");
  }
  fclose(in);

  if (ndims != 2) {
    printf("ndims is not 2 in solver.inp. this code is to generate 2D initial conditions\n");
    return(0);
  }
  if (HB != 2) {
    printf("Error: Specify \"HB\" as 2 in physics.inp.\n");
  }
	printf("Grid:\t\t\t%d X %d\n",NI,NJ);
  printf("Reference density and pressure: %lf, %lf.\n",rho_ref,p_ref);

	int i,j;
	double dx = 1000.0  / ((double)(NI-1));
	double dy = 1000.0  / ((double)(NJ-1));

	double *x, *y, *u0, *u1, *u2, *u3;
	x   = (double*) calloc (NI   , sizeof(double));
	y   = (double*) calloc (NJ   , sizeof(double));
	u0  = (double*) calloc (NI*NJ, sizeof(double));
	u1  = (double*) calloc (NI*NJ, sizeof(double));
	u2  = (double*) calloc (NI*NJ, sizeof(double));
	u3  = (double*) calloc (NI*NJ, sizeof(double));

  /* Initial perturbation center */
  double xc = 500;
  double yc = 350;
  double Cp = gamma * R / (gamma-1.0);

  /* initial perturbation parameters */
  double tc = 0.5;
  double pi = 4.0*atan(1.0);
  double rc = 250.0;

  FILE *param_rc;
  FILE *param_tc;
  param_rc = fopen("param_rc.inp", "r");
  param_tc = fopen("param_tc.inp", "r");

  fscanf(param_rc, "%lf", &rc);
  fscanf(param_tc, "%lf", &tc); 

  double T_ref = p_ref / (R * rho_ref);

	for (i = 0; i < NI; i++){
  	for (j = 0; j < NJ; j++){
	  	x[i] = i*dx;
	  	y[j] = j*dy;
      int p = NJ*i + j;

      /* temperature peturbation */
      double r      = sqrt((x[i]-xc)*(x[i]-xc)+(y[j]-yc)*(y[j]-yc));
      double dtheta = (r>rc ? 0.0 : (0.5*tc*(1.0+cos(pi*r/rc))) );
      double theta  = T_ref + dtheta;
      double Pexner = 1.0 - (grav_y*y[j])/(Cp*T_ref);

      double rho    = (p_ref/(R*theta)) * raiseto(Pexner, (1.0/(gamma-1.0)));
      double E      = rho * (R/(gamma-1.0)) * theta*Pexner;

      u0[p] = rho;
      u1[p] = 0.0;
      u2[p] = 0.0;
      u3[p] = E;
	  }
	}

  FILE *out;
  if (!strcmp(ip_file_type,"ascii")) {
    printf("Writing ASCII initial solution file initial.inp\n");
  	out = fopen("initial.inp","w");
    for (i = 0; i < NI; i++)  fprintf(out,"%lf ",x[i]);
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)  fprintf(out,"%lf ",y[j]);
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)	{
	    for (i = 0; i < NI; i++)	{
        int p = NJ*i + j;
        fprintf(out,"%lf ",u0[p]);
      }
    }
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)	{
	    for (i = 0; i < NI; i++)	{
        int p = NJ*i + j;
        fprintf(out,"%lf ",u1[p]);
      }
    }
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)	{
	    for (i = 0; i < NI; i++)	{
        int p = NJ*i + j;
        fprintf(out,"%lf ",u2[p]);
      }
    }
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)	{
	    for (i = 0; i < NI; i++)	{
        int p = NJ*i + j;
        fprintf(out,"%lf ",u3[p]);
      }
    }
    fprintf(out,"\n");
	  fclose(out);
  } else if ((!strcmp(ip_file_type,"binary")) || (!strcmp(ip_file_type,"bin"))) {
    printf("Writing binary initial solution file initial.inp\n");
  	out = fopen("initial.inp","wb");
    fwrite(x,sizeof(double),NI,out);
    fwrite(y,sizeof(double),NJ,out);
    double *U = (double*) calloc (4*NI*NJ,sizeof(double));
    for (i=0; i < NI; i++) {
      for (j = 0; j < NJ; j++) {
        int p = NJ*i + j;
        int q = NI*j + i;
        U[4*q+0] = u0[p];
        U[4*q+1] = u1[p];
        U[4*q+2] = u2[p];
        U[4*q+3] = u3[p];
      }
    }
    fwrite(U,sizeof(double),4*NI*NJ,out);
    free(U);
    fclose(out);
  }

	free(x);
	free(y);
	free(u0);
	free(u1);
	free(u2);
	free(u3);

	return(0);
}
