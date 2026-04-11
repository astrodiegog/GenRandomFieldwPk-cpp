SHELL = /usr/bin/env bash

#-- Set default include makefile
TYPE    ?= default
HOST 	?= dag

#-- Set the stuffix 
SUFFIX ?= .$(TYPE).$(HOST)

#-- Load any flags from HOST
include builds/make.host.$(HOST)

#-- Directories to: find source files, build objects, and place binary
SRC_DIRS := src
BUILD_DIR := build
BIN_DIR := bin

#-- Place all c files into a var
CPPFILES := $(foreach DIR,$(SRC_DIRS),$(wildcard $(DIR)/*.cpp))

#-- Prepend BUILD_DIR and appends .o to all source C++ files
CLEAN_OBJS   := $(CPPFILES:%=$(BUILD_DIR)/%.o)

#-- In the case any CPPFILES changes
OBJS   := $(CPPFILES:%=$(BUILD_DIR)/%.o)

#-- Set default compiler and flags
CXX               ?= gcc
CXXFLAGS_OPTIMIZE ?= -g -Ofast
CXXFLAGS_DEBUG    ?= -g -O0 -Wall

BUILD             ?= DEBUG

#-- Grab the appropriate CPPFLAGS
CXXFLAGS          += $(CXXFLAGS_$(BUILD))

#-- Add flags and libraries as needed
CXXFLAGS += $(DFLAGS) -Isrc

CXXFLAGS += -I$(HDF5_ROOT)/include
CXXFLAGS += -I$(FFTW_ROOT)/include

LIBS     = -L$(HDF5_ROOT)/lib -lhdf5
LIBS   	 += -L$(FFTW_ROOT)/lib -lfftw3_mpi -lfftw3 -lm
LIBS 	 += -L$(MPI_ROOT)/lib -lmpi

#-- Define the executable
EXEC := bin/PkField$(SUFFIX)

#-- Label flags
MACRO_FLAGS := -DMACRO_FLAGS='"$(DFLAGS)"'
DFLAGS      += $(MACRO_FLAGS)

#-- Define the link editor & flags
LD := $(CXX)
LDFLAGS := $(CXXFLAGS)




#-- Execute the target executable once prereq-build is set
$(EXEC): prereq-build $(OBJS)
	echo "We are linking tings!"
	echo $(LDFLAGS)
	mkdir -p bin/ && $(LD) $(LDFLAGS) $(OBJS) -o $(EXEC) $(LIBS)

#-- Build step for the C source code
$(BUILD_DIR)/%.cpp.o: %.cpp
	echo "We are building object files!"
	mkdir -p $(dir $@)
	echo $(CXX)
	$(CXX) $(CXXFLAGS) -c $< -o $@


.PHONY: clean

all: 
	echo $(DFLAGS)
	echo $(CPPFILES)
	echo $(OBJS)
	echo $(OBJS1)


prereq-build:
	echo "We are in prereq-build!"
	builds/prereq.sh build $(HOST)


clean:
	rm -f $(CLEAN_OBJS)
	rm -r $(BUILD_DIR)

