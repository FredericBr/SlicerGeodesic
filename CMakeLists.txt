cmake_minimum_required(VERSION 3.5)

project(GeodesicSlicer)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Modules/GeodesicSlicer")
set(EXTENSION_CATEGORY "Informatics")
set(EXTENSION_CONTRIBUTORS "Frederic Briend (UNICAEN, University of Tours), Antoine Nourry (UMS 3408)")
set(EXTENSION_DESCRIPTION "This module calculates geodesic path in 3D structure. Thanks to this geodesic path, this module can draw an EEG 10-20 system, determine the projected scalp stimulation site and correct the rTMS resting motor threshold by correction factor.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/FredericBr/SlicerGeodesic/master/GeodesicSlicer.png")
set(EXTENSION_SCREENSHOTURLS "https://www.slicer.org/w/img_auth.php/c/ce/Screen-shot_of_the_GeodesicSlicer_program._.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(GeodesicSlicer)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
