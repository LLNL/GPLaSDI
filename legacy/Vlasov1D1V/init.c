#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


int main()
{  
  double pi = 4.0 * atan(1.0);
  double two_pi = 8.0 * atan(1.0);

  double L = two_pi;
  double T = 1.0;
  double k = 1.0;
  
  FILE *param_T;
  FILE *param_k;
  param_T = fopen("param_T.inp", "r");
  param_k = fopen("param_k.inp", "r");

  fscanf(param_T, "%lf", &T);
  fscanf(param_k, "%lf", &k);

  int NI,NJ,ndims,n_iter;
  FILE *in;

  char ip_file_type[50];
  strcpy(ip_file_type,"ascii");

  printf("Reading file \"solver.inp\"...\n");
  in = fopen("solver.inp","r");
  if (!in) {

    fprintf(stderr,"Error: Input file \"solver.inp\" not found.\n");
    return(0);

  } else {

    char word[500];
    fscanf(in,"%s",word);
    if (!strcmp(word, "begin")){
      while (strcmp(word, "end")){
        fscanf(in,"%s",word);
        if (!strcmp(word, "ndims")) fscanf(in,"%d",&ndims);
        else if (!strcmp(word, "size")) {
          fscanf(in,"%d",&NI);
          fscanf(in,"%d",&NJ);
        } else if (!strcmp(word, "ip_file_type")) {
           fscanf(in,"%s",ip_file_type);
        }
      }

    } else {

      fprintf(stderr,"Error: Illegal format in solver.inp. Crash and burn!\n");
      return(0);

    }

  }

  fclose(in);

  if (ndims != 2) {
    fprintf(stderr,"ndims is not 2 in solver.inp. this code is to generate 2D initial conditions\n");
    return(0);
  }
  printf("Grid: %d, %d\n",NI,NJ);

  int i,j;
  double dx = two_pi / ((double)NI);
  double dv = 14.0 / ((double)NJ);
  double start_x = 0.0;
  double start_v = -7.0;

  double *x, *v, *f;
  x = (double*) calloc (NI   , sizeof(double));
  v = (double*) calloc (NJ   , sizeof(double));
  f = (double*) calloc (NI*NJ, sizeof(double));

  for (i = 0; i < NI; i++){
    for (j = 0; j < NJ; j++){
      x[i] = start_x + i*dx;
      v[j] = start_v + j*dv;
      int p = NJ*i + j;
      double temp1 = cos(2.0 * k * pi * x[i] / L);
      double temp2 = exp(- (v[j] -2.0) * (v[j] - 2.0) / (2.0 * T));
      double temp3 = exp(- (v[j] + 2.0) * (v[j] + 2.0) / (2.0 * T));
      double temp4 = sqrt(2.0 * pi * T);
      f[p] = 16.0 * (1.0 + 0.1 * temp1) * 0.5 * (temp2 / temp4 + temp3 / temp4);
      //f[p] = 16.*(1. + 0.1*cos(2.*k*pi*x[i]/L))*(exp(-((v[j]-2.)*(v[j]-2.))/(2.*T))/sqrt(2.*pi*T) + exp(-((v[j]+2.)*(v[j]+2.))/(2.*T))/sqrt(2.*pi*T))/2.;
    }
  }

  FILE *out;
  if (!strcmp(ip_file_type,"ascii")) {

    out = fopen("initial.inp","w");
    for (i = 0; i < NI; i++)  fprintf(out,"%1.16e ",x[i]);
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)  fprintf(out,"%1.16e ",v[j]);
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)	{
       for (i = 0; i < NI; i++)	{
          int p = NJ*i + j;
          fprintf(out,"%1.16e ", f[p]);
       }
    }
    fprintf(out,"\n");
    fclose(out);

  } else if ((!strcmp(ip_file_type,"binary")) || (!strcmp(ip_file_type,"bin"))) {

    printf("Writing binary exact solution file initial.inp\n");
    out = fopen("initial.inp","wb");
    fwrite(x, sizeof(double),NI,out);
    fwrite(v, sizeof(double),NJ,out);
    double *F = (double*) calloc (NI*NJ,sizeof(double));
    for (i=0; i < NI; i++) {
      for (j = 0; j < NJ; j++) {
        int p = NJ*i + j;
        int q = NI*j + i;
        F[q+0] = f[p];
      }
    }
    fwrite(F, sizeof(double),NI*NJ,out);
    free(F);
    fclose(out);

  }

  free(x);
  free(v);
  free(f);

  return(0);
}
