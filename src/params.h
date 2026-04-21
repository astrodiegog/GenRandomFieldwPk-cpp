#include "HDF5_utils.h"
#pragma once

struct PS_Params
{
	std::uint_fast32_t seed;

    int ndims;
    double Lbox;
    int Ng;

    float As, ks, ns;

};

/*\fn void Parse_Params(char *, struct* PS_Params) */
/*! Routine to place information from the parameter file onto the PS_Params struct */
extern void Parse_Params(char *param_file, struct PS_Params *ps_params);

/*\fn void Parse_Param(char *, char *, struct* PS_Params) */
/*! Routine to place key-value pairsonto PS_Params struct */
extern void Parse_Param(char *key, char *value, struct PS_Params *ps_params);


