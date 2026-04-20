#include <string.h>

#include <math.h>

#include <complex.h>
#include "hdf5.h"
#include <fftw3.h>

/* Define max string length */
#define MAXLEN 1024

/* \fn void Write_HDF5_1Dgroup(hid_t, char *, hid_t, double *) */
/* Routine to write array of double */
extern void Write_HDF5_1Dgroup(hid_t grp_id, char *arr_name, hid_t dataspace_id, double *data_arr);


/* \fn void Write_FFTWarr_1Dgroup(hid_t, char *, hid_t, fftw_complex *, int) */
/* Routine to write Real&Imaginary array of fftw_complex with size Nx */
extern void Write_FFTWarr_1Dgroup(hid_t grp_id, char *arr_prefix, hid_t dataspace_id, fftw_complex *FFTW_arr, int Nx);

/* \fn void Write_FFTWarr_2Dgroup(hid_t, char *, hid_t, fftw_complex *, int, int) */
/* Routine to write Real&Imaginary array of fftw_complex with size Nx,Ny */
extern void Write_FFTWarr_2Dgroup(hid_t grp_id, char *arr_prefix, hid_t dataspace_id, fftw_complex *FFTW_arr, int Nx, int Ny);

/* \fn void Write_FFTWarr_3Dgroup(hid_t, char *, hid_t, fftw_complex *, int, int, int) */
/* Routine to write Real&Imaginary array of fftw_complex with size Nx,Ny,Nz */
extern void Write_FFTWarr_3Dgroup(hid_t grp_id, char *arr_prefix, hid_t dataspace_id, fftw_complex *FFTW_arr, int Nx, int Ny, int Nz);


/* \fn void Write_HDF5_int_attribute(hid_t, char *, hid_t, int *) */
/* Routine to write array of ints as attribute */
extern void Write_HDF5_int_attribute(hid_t grp_id, char *arr_name, hid_t dataspace_id, int *attr_int_arr);

/* \fn void Write_HDF5_double_attribute(hid_t, char *, hid_t, double *) */
/* Routine to write array of double as attribute */
extern void Write_HDF5_double_attribute(hid_t grp_id, char *arr_name, hid_t dataspace_id, double *attr_double_arr);


/* \fn void Write_HDF5_dataset(hid_t, char *, hid_t, double *) */
/* Routine to write double dataset of arr_name to grp_id */
extern void Write_HDF5_dataset(hid_t grp_id, char *arr_name, hid_t dataspace_id, double *data_arr);

/*! \fn void Write_HDF5_longint_dataset(hid_t, char *, hid_t, long int*) */
/* Routine to write a long integer dataset of arr_name to grp_id */
extern void Write_HDF5_longint_dataset(hid_t grp_id, char *arr_name, hid_t dataspace_id, long int *data_arr);

