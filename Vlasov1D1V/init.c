#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


int main()
{  
  double pi = 4.0 * atan(1.0);
  double two_pi = 8.0 * atan(1.0);

  double k_Dr = 1.0;
  int    N_mode = 1;
//  double L = two_pi;
  double T1 = 1.0;
  double T2 = 1.0;
  double eps = 0.1;
  double v_d = 2.0;

//  FILE *param_T;
//  FILE *param_k;
//  param_T1 = fopen("param_T1.inp", "r");
//  param_T2 = fopen("param_T2.inp", "r");
//  param_k_Dr = fopen("param_k_Dr.inp", "r");
//  param_eps = fopen("param_eps.inp", "r");
//  param_V_d = fopen("param_V_d.inp", "r");

//  fscanf(param_T, "%lf", &T);
//  fscanf(param_k, "%lf", &k);

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
        }
        else if (!strcmp(word, "ip_file_type")) {
           fscanf(in,"%s",ip_file_type);
        }
        else if (!strcmp(word, "v_drift")) {
           fscanf(in,"%lf",&v_d);
        }
        else if (!strcmp(word, "T1")) {
           fscanf(in,"%lf",&T1);
        }
        else if (!strcmp(word, "T2")) {
           fscanf(in,"%lf",&T2);
        }
        else if (!strcmp(word, "eps")) {
           fscanf(in,"%lf",&eps);
        }
        else if (!strcmp(word, "k_Dr")) {
           fscanf(in,"%lf",&k_Dr);
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
  printf("v_drift: %f\n",v_d);
  printf("T1: %f\n",T1);
  printf("T2: %f\n",T2);
  printf("eps: %f\n",eps);
  printf("k_Dr: %f\n",k_Dr);

  int i,j;
  double L_box = two_pi * N_mode / k_Dr;
  double dx = L_box / ((double)NI);
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
      double temp1 = cos(2.0 * k_Dr * pi * x[i] / L_box);
      double temp2 = exp(- (v[j] - v_d) * (v[j] - v_d) / (2.0 * T1));
      double temp3 = exp(- (v[j] + v_d) * (v[j] + v_d) / (2.0 * T2));
      double temp4 = sqrt(2.0 * pi * T1);
      double temp5 = sqrt(2.0 * pi * T2);
      f[p] = 8.0 * (1.0 + eps * temp1) * (temp2 / temp4 + temp3 / temp5);
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
