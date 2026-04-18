#include "HDF5_utils.h"

extern void Write_HDF5_int_attribute(hid_t grp_id, char *arr_name, hid_t dataspace_id, int *attr_int_arr)
{
    hid_t attr_id;
    herr_t status;

    attr_id = H5Acreate(grp_id, arr_name, H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_INT, attr_int_arr);
    status = H5Aclose(attr_id);

    return;
}


extern void Write_HDF5_double_attribute(hid_t grp_id, char *arr_name, hid_t dataspace_id, double *attr_double_arr)
{
    hid_t attr_id;
    herr_t status;

    attr_id = H5Acreate(grp_id, arr_name, H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_DOUBLE, attr_double_arr);
    status = H5Aclose(attr_id);

    return;
}


extern void Write_HDF5_dataset(hid_t grp_id, char *arr_name, hid_t dataspace_id, double *data_arr)
{
	hid_t dataset_id;
    herr_t status;

    dataset_id = H5Dcreate(grp_id, arr_name, H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_arr);
    status = H5Dclose(dataset_id);

    return;
}


extern void Write_FFTWarr_1Dgroup(hid_t grp_id, char *arr_prefix, hid_t dataspace_id, fftw_complex *FFTW_arr, int Nx)
{
	int i;
    double FFTW_arr_Real[Nx], FFTW_arr_Imag[Nx];
	char realarr_name[MAXLEN], imagarr_name[MAXLEN];

    for (i = 0; i < Nx; i++)
    {
        FFTW_arr_Real[i] = FFTW_arr[i][0];
        FFTW_arr_Imag[i] = FFTW_arr[i][1];
    }

	strcpy(realarr_name, arr_prefix);
	strcpy(imagarr_name, arr_prefix);

	strcat(realarr_name, "_Real");
	strcat(imagarr_name, "_Imag");

	Write_HDF5_dataset(grp_id, realarr_name, dataspace_id, &FFTW_arr_Real[0]);
    Write_HDF5_dataset(grp_id, imagarr_name, dataspace_id, &FFTW_arr_Imag[0]);

    return;
}


extern void Write_FFTWarr_2Dgroup(hid_t grp_id, char *arr_prefix, hid_t dataspace_id, fftw_complex *FFTW_arr, int Nx, int Ny)
{
    int i, j, indx;
    double FFTW_arr_Real[Nx * Ny], FFTW_arr_Imag[Nx * Ny];
    char realarr_name[MAXLEN], imagarr_name[MAXLEN];

    for (i = 0; i < Nx; i++)
    {
        for (j = 0; j < Ny; j++)
        {
            indx = j + Ny * i;
            FFTW_arr_Real[indx] = FFTW_arr[indx][0];
            FFTW_arr_Imag[indx] = FFTW_arr[indx][1];
        }
    }

    strcpy(realarr_name, arr_prefix);
    strcpy(imagarr_name, arr_prefix);

    strcat(realarr_name, "_Real");
    strcat(imagarr_name, "_Imag");

    Write_HDF5_dataset(grp_id, realarr_name, dataspace_id, &FFTW_arr_Real[0]);
    Write_HDF5_dataset(grp_id, imagarr_name, dataspace_id, &FFTW_arr_Imag[0]);

    return;
}

