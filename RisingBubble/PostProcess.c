/*
  Rising Thermal Bubble:-
  The code takes a binary solution file (that contains the 
  conserved variable (rho, rho*u, rho*v, e) as its input 
  and calculates the primitive atmospheric flow variables:
  rho, u, v, P, theta, pi, rho0, P0, theta0, pi0
  and writes them to a tecplot file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct _parameters_{
  double grav_x, grav_y, R, gamma, P_ref, rho_ref;
  int HB;
} Parameters;

void IncrementFilename(char *f)
{
  if (f[7] == '9') {
    f[7] = '0';
    if (f[6] == '9') {
      f[6] = '0';
      if (f[5] == '9') {
        f[5] = '0';
        if (f[4] == '9') {
          f[4] = '0';
          if (f[3] == '9') {
            f[3] = '0';
            fprintf(stderr,"Warning: file increment hit max limit. Resetting to zero.\n");
          } else {
            f[3]++;
          }
        } else {
          f[4]++;
        }
      } else {
        f[5]++;
      }
    } else {
      f[6]++;
    }
  } else {
    f[7]++;
  }
}

double raiseto(double x, double a)
{
  return(exp(a*log(x)));
}

void WriteTecplot2D(int nvars,int imax, int jmax,double *x,double *u,char *f)
{
  printf("\tWriting tecplot solution file %s.\n",f);
  FILE *out;
  out = fopen(f,"w");
  if (!out) {
    fprintf(stderr,"Error: could not open %s for writing.\n",f);
    return;
  }

  double *X = x;
  double *Y = x+imax;

  /* writing tecplot data file headers */
  fprintf(out,"VARIABLES=\"I\",\"J\",\"X\",\"Y\",");
  fprintf(out,"\"RHO\",\"U\",\"V\",\"P\",");
  fprintf(out,"\"THETA\",\"RHO0\",\"P0\",");
  fprintf(out,"\"PI0\",\"THETA0\",\n");
  fprintf(out,"ZONE I=%d,J=%d,F=POINT\n",imax,jmax);

  /* writing the data */
  int i,j;
  for (j=0; j<jmax; j++) {
    for (i=0; i<imax; i++) {
      int v, p = i + imax*j;
      fprintf(out,"%4d %4d ",i,j);
      fprintf(out,"%1.16E %1.16E ",X[i],Y[j]);
      for (v=0; v<nvars; v++) fprintf(out,"%1.16E ",u[nvars*p+v]);
      fprintf(out,"\n");
    }
  }
  fclose(out);
  return;
}

void WriteText2D(int nvars,int imax, int jmax,double *x,double *u,char *f)
{
  printf("\tWriting text solution file %s.\n",f);
  FILE *out;
  out = fopen(f,"w");
  if (!out) {
    fprintf(stderr,"Error: could not open %s for writing.\n",f);
    return;
  }

  double *X = x;
  double *Y = x+imax;

  /* writing the data */
  int i,j;
  for (j=0; j<jmax; j++) {
    for (i=0; i<imax; i++) {
      int v, p = i + imax*j;
      fprintf(out,"%4d %4d ",i,j);
      fprintf(out,"%1.16E %1.16E ",X[i],Y[j]);
      for (v=0; v<nvars; v++) fprintf(out,"%1.16E ",u[nvars*p+v]);
      fprintf(out,"\n");
    }
  }
  fclose(out);
  return;
}

int PostProcess(char *fname, char *oname, void *p, int flag)
{
  Parameters *params = (Parameters*) p;
  FILE *in; in = fopen(fname,"rb");

  if (!in) return(-1);

  printf("Reading file %s.\n",fname);
  int ndims, nvars;
  double *U,*x;

  /* read the file headers */
  fread(&ndims,sizeof(int),1,in);
  fread(&nvars,sizeof(int),1,in);

  /* some checks */
  if (ndims != 2) {
    printf("Error: ndims in %s not equal to 2!\n",fname);
    return(1);
  }
  if (nvars != 4) {
    printf("Error: nvars in %s not equal to 4!\n",fname);
    return(1);
  }

  /* read dimensions */
  int dims[ndims];
  fread(dims,sizeof(int),ndims,in);
  printf("Dimensions: %d x %d\n",dims[0],dims[1]);
  printf("Nvars     : %d\n",nvars);

  /* allocate grid and solution arrays */
  x = (double*) calloc (dims[0]+dims[1]       ,sizeof(double));
  U = (double*) calloc (dims[0]*dims[1]*nvars ,sizeof(double));

  /* read grid and solution */
  fread(x,sizeof(double),dims[0]+dims[1]      ,in);
  fread(U,sizeof(double),dims[0]*dims[1]*nvars,in);
  /* done reading */
  fclose(in);

  int imax = dims[0];
  int jmax = dims[1];

  /* allocate primitive variable array (rho, u, v, P, theta, rho0, P0, pi0, theta0) */
  int evars = 5;
  double *Q = (double*) calloc ((nvars+evars)*imax*jmax,sizeof(double));
    
  /* calculate primitive variables */
  int i, j;
  double *X           = x;
  double *Y           = x+imax;
  double grav_y       = params->grav_y;
  double R            = params->R;
  double gamma        = params->gamma;
  double P_ref        = params->P_ref;
  double rho_ref      = params->rho_ref;
  double T_ref        = P_ref / (R*rho_ref);
  double inv_gamma_m1 = 1.0 / (gamma-1.0);
  double Cp           = gamma * inv_gamma_m1 * R;

  for (i=0; i<imax; i++) {
    for (j=0; j<jmax; j++) {
      int p = i + imax*j;

      double rho0, theta0, Pexner, P0;
      theta0  = T_ref;
      Pexner  = 1.0 - (grav_y*Y[j])/(Cp*T_ref);
      rho0    = (P_ref/(R*theta0)) * raiseto(Pexner, inv_gamma_m1);
      P0      = P_ref   * raiseto(Pexner, gamma*inv_gamma_m1);

      double rho, uvel, vvel, E, P, theta;
      rho   = U[nvars*p+0];
      uvel  = U[nvars*p+1] / rho;
      vvel  = U[nvars*p+2] / rho;
      E     = U[nvars*p+3];
      P     = (gamma-1.0) * (E - 0.5*rho*(uvel*uvel+vvel*vvel));
      theta = (E-0.5*rho*(uvel*uvel+vvel*vvel))/(Pexner*rho) * ((gamma-1.0)/R);

      Q[(nvars+evars)*p+0] = rho;
      Q[(nvars+evars)*p+1] = uvel;
      Q[(nvars+evars)*p+2] = vvel;
      Q[(nvars+evars)*p+3] = P;
      Q[(nvars+evars)*p+4] = theta;
      Q[(nvars+evars)*p+5] = rho0;
      Q[(nvars+evars)*p+6] = P0;
      Q[(nvars+evars)*p+7] = Pexner;
      Q[(nvars+evars)*p+8] = theta0;
    }
  }

  /* write Tecplot/Text file */
  if (flag) WriteTecplot2D(nvars+evars,imax,jmax,x,Q,oname);
  else      WriteText2D   (nvars+evars,imax,jmax,x,Q,oname);

  /* clean up */
  free(U);
  free(Q);
  free(x);
}

int main()
{
  FILE *out1, *out2, *in, *inputs;
  char filename[50], op_file_format[50], tecfile[50], overwrite[50];
  
  /*
  int flag;
  printf("Write tecplot file (1) or plain text file (0): ");
  scanf("%d",&flag);
  */

  int flag = 0;
 
  if ((flag != 1) && (flag != 0)) {
    printf("Error: Invalid input. Should be 1 or 0.\n");
    return(0);
  }

  printf("Reading solver.inp.\n");
  inputs = fopen("solver.inp","r");
  if (!inputs) {
    fprintf(stderr,"Error: File \"solver.inp\" not found.\n");
    return(1);
  } else {
	  char word[100];
    fscanf(inputs,"%s",word);
    if (!strcmp(word, "begin")){
	    while (strcmp(word, "end")){
		    fscanf(inputs,"%s",word);
   			if      (!strcmp(word, "op_file_format"   ))  fscanf(inputs,"%s" ,op_file_format);
   			else if (!strcmp(word, "op_overwrite"     ))  fscanf(inputs,"%s" ,overwrite      );
      }
    }
    fclose(inputs);
  }
  if (strcmp(op_file_format,"binary") && strcmp(op_file_format,"bin")) {
    printf("Error: solution output needs to be in binary files.\n");
    return(0);
  }

  Parameters params;
  /* default values */
  params.grav_x   = 0.0;
  params.grav_y   = 9.8;
  params.R        = 287.058;
  params.gamma    = 1.4;
  params.P_ref    = 100000.0;
  params.rho_ref  = 100000.0 / (params.R * 300.0);
  params.HB       = 0;
  /* read these parameters from file */
  printf("Reading physics.inp.\n");
  inputs = fopen("physics.inp","r");
  if (!inputs) {
    fprintf(stderr,"Error: File \"physics.inp\" not found.\n");
    return(1);
  } else {
	  char word[100];
    fscanf(inputs,"%s",word);
    if (!strcmp(word, "begin")){
	    while (strcmp(word, "end")){
		    fscanf(inputs,"%s",word);
   			if      (!strcmp(word, "gamma"))    fscanf(inputs,"%lf",&params.gamma);
   			else if (!strcmp(word, "gravity")) {
          fscanf(inputs,"%lf",&params.grav_x);
          fscanf(inputs,"%lf",&params.grav_y);
        } else if (!strcmp(word,"p_ref"))   fscanf(inputs,"%lf",&params.P_ref);
        else if   (!strcmp(word,"rho_ref")) fscanf(inputs,"%lf",&params.rho_ref);
        else if   (!strcmp(word,"HB"))      fscanf(inputs,"%d",&params.HB);
      }
    }
    fclose(inputs);
  }
  if (params.HB != 2) {
    printf("Error: \"HB\" must be specified as 2 in physics.inp.\n");
    return(0);
  }

  if (!strcmp(overwrite,"no")) {
    strcpy(filename,"op_00000.bin");
    while(1) {
      /* set filename */
      strcpy(tecfile,filename);
      tecfile[9]  = 'd';
      tecfile[10] = 'a';
      tecfile[11] = 't';
      int err = PostProcess(filename, tecfile, &params, flag);
      if (err == -1) {
        printf("No more files found. Exiting.\n");
        break;
      }
      IncrementFilename(filename);
    }
  } else if (!strcmp(overwrite,"yes")) {
    strcpy(filename,"op.bin");
    /* set filename */
    strcpy(tecfile,filename);
    tecfile[3] = 'd';
    tecfile[4] = 'a';
    tecfile[5] = 't';
    int err = PostProcess(filename, tecfile, &params, flag);
    if (err == -1) {
      printf("Error: op.bin not found.\n");
      return(0);
    }
  }

  return(0);
}
