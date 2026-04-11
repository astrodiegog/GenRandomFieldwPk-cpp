#!/bin/bash

if [ "$ENVSET" == "1" ]; then
  exit 0
fi

if [ "$1" == "build" ]; then

  case $2 in
    lux)
      if ! module is-loaded gcc; then
        echo "modulefile required: gcc"
        echo "do: 'module load gcc hdf5/1.14.4-parallel openmpi/4.1.5'"
        exit 1
      fi
      ;;
    dag)
       if ! brew list gcc ; then
         echo "do: 'brew install gcc'"
         exit 1
       fi
     ;;
  esac

fi

exit 0
