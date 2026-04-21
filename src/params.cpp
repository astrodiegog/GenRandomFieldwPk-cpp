#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "params.h"

#include <dirent.h>

extern void Parse_Param(char *key, char *value, struct PS_Params *ps_params)
{
	if (!strcmp(key, "ndims")){
		ps_params->ndims = atoi(value);
	}
	else if (!strcmp(key, "seed")) {
		ps_params->seed = atoi(value);
	}
    else if (!strcmp(key, "Lbox")) {
        ps_params->Lbox = (double) atof(value);
    }
    else if (!strcmp(key, "Ng")) {
        ps_params->Ng = atoi(value);
    }
	else if (!strcmp(key, "As")) {
        ps_params->As = (double) atof(value);
    }
	else if (!strcmp(key, "ks")) {
        ps_params->ks = (double) atof(value);
    }
	else if (!strcmp(key, "ns")) {
        ps_params->ns = (double) atof(value);
    }
    else
    {
        printf("UNKNOWN PARAMETER: %s = %s\n", key, value);
    }
}

extern void Parse_Params(char *param_file, struct PS_Params *ps_params)
{
    /* Declare buffer array*/
    char buff[MAXLEN]; // define maxlen in header file

    /* Declare pointer to pointer array */
    char *str;

    /* Declare pointer to parameter file */
    FILE *fptr;

    /* Key-Value pairs to read into param values */
    char key[MAXLEN], value[MAXLEN];

	/* Open the file */
	fptr = fopen(param_file, "r");
	
    if (fptr == NULL)
    {
        printf("Unable to open parameter file.... oopsies ! \n");
    }

    /* Place a max of MAXLEN number of character into buff from fptr*/
    while ( (str = fgets(buff, MAXLEN, fptr) ) )
    {
        /* Place token onto buff string with = delimiter */
        str = strtok(buff, "=");

        /* Grab the key */
        if (str)
        {
            /* Copy token str onto key array */
            strncpy(key, str, MAXLEN);
        }
        else
        {
            continue;
        }

        /* Continue tokenizing */
        str = strtok(NULL, "=");

        /* Grab the value */
        if (str)
        {
            /* Copy token str onto key array */
            strncpy(value, str, MAXLEN);
        }
        else
        {
            continue;
        }

        /* Place key-value char* onto the PS_Param struct */
        Parse_Param(key, value, ps_params);

    }

    fclose(fptr);

}



