# coding=utf-8

# Copyright :
# - EA7466 CHU Caen - Université de Caen Normandie,
# - UMS 3408 CHU Caen - CNRS,
# contributeurs : Frédéric Briend, Olivier Etard, Antoine Nourry, Nicolas Delcroix.
# 01/07/2018.

# briend@cyceron.fr ; olivier.etard@unicaen.fr

# Ce logiciel est un programme informatique servant à [rappeler les caractéristiques techniques de votre logiciel]. 

# Ce logiciel est régi par la licence CeCILL soumise au droit français et respectant les principes de diffusion des logiciels libres. Vous pouvez
# utiliser, modifier et/ou redistribuer ce programme sous les conditions de la licence CeCILL telle que diffusée par le CEA, le CNRS et l'INRIA 
# sur le site "http://www.cecill.info".

# En contrepartie de l'accessibilité au code source et des droits de copie, de modification et de redistribution accordés par cette licence, il n'est
# offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons, seule une responsabilité restreinte pèse sur l'auteur du programme,  le
# titulaire des droits patrimoniaux et les concédants successifs.

# A cet égard  l'attention de l'utilisateur est attirée sur les risques associés au chargement,  à l'utilisation,  à la modification et/ou au
# développement et à la reproduction du logiciel par l'utilisateur étant donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
# manipuler et qui le réserve donc à des développeurs et des professionnels avertis possédant  des  connaissances  informatiques approfondies.  Les
# utilisateurs sont donc invités à charger  et  tester  l'adéquation  du logiciel à leurs besoins dans des conditions permettant d'assurer la
# sécurité de leurs systèmes et ou de leurs données et, plus généralement, à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

# Le fait que vous puissiez accéder à cet en-tête signifie que vous avez pris connaissance de la licence CeCILL, et que vous en avez accepté les termes.
# ********************************************************************************************
# Copyright:
# - EA7466 CHU Caen - Université de Caen Normandie,
# - UMS 3408 CHU Caen - CNRS,
# contributors: Frédéric Briend, Olivier Etard, Antoine Nourry, Nicolas Delcroix.
# [date of creation].

# briend@cyceron.fr ; olivier.etard@unicaen.fr

# This software is a computer program whose purpose is to [describe functionalities and technical features of your software].

# This software is governed by the CeCILL license under French law and abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 

# As a counterpart to the access to the source code and  rights to copy, modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the economic rights,  and the successive licensors  have only  limited
# liability. 

# In this respect, the user's attention is drawn to the risks associated with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software, that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the same conditions as regards security. 

# The fact that you are presently reading this means that you have had knowledge of the CeCILL license and that you accept its terms.
# ********************************************************************************************

import os
import unittest
import vtk, qt, ctk, slicer
import logging, time
import numpy, math, slicer, math
from slicer.ScriptedLoadableModule import *
from numpy import mean
from heapq import nsmallest

#
# GeodesicSlicer
#

class GeodesicSlicer(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Geodesic Slicer" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Informatics"]
    self.parent.dependencies = []
    self.parent.contributors = ["Frederic Briend (ISTS, UNICAEN), Antoine Nourry (UMS 3408, UNICAEN), Nicolas Delcroix (UMS 3408, UNICAEN), Olivier Etard (ISTS, UNICAEN, CHU Caen)"]
    self.parent.helpText = """
This module calculates geodesic path in 3D structure. Thanks to this geodesic path, this module can draw an EEG 10-20 system, 
determine the projected scalp stimulation site and correct the rTMS resting motor threshold by correction factor.
<p><li>See <a href="https://www.slicer.org/wiki/Documentation/Nightly/Modules/GeodesicSlicer">Geodesic Slicer documentation</a> for more details.</li>
"""
    #self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This module was developed by Frederic Briend and Antoine Nourry. Thank you to Andras Lasso, Csaba Pinter, Clement Nathou, Olivier Etard and Sonia Dollfus.
<p>GeodesicSlicer is a research tool. It may not be accurate, use it at your own risk.
<p>This work was supported by CHU Caen, Region Normandie and UNICAEN. If you use this module, please cite the following article:
<li>* Briend F. et al., Personalized or standardized target for the treatment of auditory hallucinations by rTMS? Brain Stimulation, submitted</li>
""" # replace with organization, grant and thanks.

#
# GeodesicSlicerWidget
#

class GeodesicSlicerWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """  
  def setup(self):
	ScriptedLoadableModuleWidget.setup(self)
	
	#####
	## Create a mesh area
	#####
	self.meshCollapsibleButton = ctk.ctkCollapsibleButton()
	self.meshCollapsibleButton.text = "Create a mesh"
	self.layout.addWidget(self.meshCollapsibleButton)
	
	# Layout within the sample collapsible button
	self.meshFormLayout = qt.QFormLayout(self.meshCollapsibleButton)
	
	# The Input Volume Selector
	self.inputFrame = qt.QFrame(self.meshCollapsibleButton)
	self.inputFrame.setLayout(qt.QHBoxLayout())
	self.meshFormLayout.addRow(self.inputFrame)
	self.inputSelector = qt.QLabel("Input Volume.nii: ", self.inputFrame)
	self.inputFrame.layout().addWidget(self.inputSelector)
	self.inputSelector = slicer.qMRMLNodeComboBox(self.inputFrame)
	self.inputSelector.nodeTypes = ( ("vtkMRMLScalarVolumeNode"), "" )
	self.inputSelector.addEnabled = False
	self.inputSelector.removeEnabled = False
	self.inputSelector.setMRMLScene( slicer.mrmlScene )
	self.inputFrame.layout().addWidget(self.inputSelector)
	
	# Apply Button 'Create a quick mesh'
	self.applyButton2 = qt.QPushButton("Create a quick mesh")
	self.applyButton2.toolTip = "Run the algorithm."
	self.applyButton2.enabled = True
	self.inputFrame.layout().addWidget(self.applyButton2)
	
	# connections 'Create a quick mesh'
	self.applyButton2.connect('clicked(bool)', self.onApplyButton2)
	
	# Apply Button 'Create a mesh'
	self.applyButtonA = qt.QPushButton("Create a mesh")
	self.applyButtonA.toolTip = "Run the algorithm with filling holes smoothing (better, but longer)."
	self.applyButtonA.enabled = True
	self.inputFrame.layout().addWidget(self.applyButtonA)
	
	# connections 'Create a mesh'
	self.applyButtonA.connect('clicked(bool)', self.onApplyButtonA)
	
	#####
	##Shortest Path area
	#####
	self.parametersCollapsibleButton = ctk.ctkCollapsibleButton()
	self.parametersCollapsibleButton.text = "Parameters to find the shortest path"
	self.layout.addWidget(self.parametersCollapsibleButton)
	self.parametersCollapsibleButton.collapsed = True

	# Layout within the dummy collapsible button
	parametersFormLayout = qt.QFormLayout(self.parametersCollapsibleButton)

	# Source points (vtkMRMLMarkupsFiducialNode)
	self.SourceSelector = slicer.qMRMLNodeComboBox()
	self.SourceSelector.nodeTypes = ( ("vtkMRMLMarkupsFiducialNode"), "" )
	self.SourceSelector.addEnabled = True
	self.SourceSelector.removeEnabled = False
	self.SourceSelector.noneEnabled = True
	self.SourceSelector.showHidden = False
	self.SourceSelector.renameEnabled = True
	self.SourceSelector.showChildNodeTypes = False
	self.SourceSelector.setMRMLScene( slicer.mrmlScene )
	self.SourceSelector.setToolTip( "Pick up a Markups node listing fiducials. The name must be different to model name" )
	parametersFormLayout.addRow("Source points: ", self.SourceSelector)

	# input model	
	self.inputTargetModelSelector = slicer.qMRMLNodeComboBox()
	self.inputTargetModelSelector.nodeTypes = ( ("vtkMRMLModelNode"), "" )
	self.inputTargetModelSelector.selectNodeUponCreation = True
	self.inputTargetModelSelector.addEnabled = False
	self.inputTargetModelSelector.removeEnabled = False
	self.inputTargetModelSelector.noneEnabled = False
	self.inputTargetModelSelector.showHidden = False
	self.inputTargetModelSelector.showChildNodeTypes = False
	self.inputTargetModelSelector.setMRMLScene( slicer.mrmlScene )
	self.inputTargetModelSelector.setToolTip( "Select the model the other will be transformed to. This model required to contain a dense set of points." )
	parametersFormLayout.addRow("Input STL model: ", self.inputTargetModelSelector)

	# Apply Button "Find the shortest path"
	self.applyButton = qt.QPushButton("Find the shortest path")
	self.applyButton.toolTip = "Run the Dijkstra's algorithm for finding the shortest path"
	#self.applyButton.enabled = True
	parametersFormLayout.addRow(self.applyButton)

	# connections "Find the shortest path"
	self.applyButton.connect('clicked(bool)', self.onApplyButton)
	
	# Apply Button "Draw the shortest path"
	self.applyButtonDraw = qt.QPushButton("Draw the shortest path")
	self.applyButtonDraw.toolTip = "Draw the Dijkstra's algorithm for finding the shortest path"
	#self.applyButtonDraw.enabled = True
	parametersFormLayout.addRow(self.applyButtonDraw)

	# connections "Draw the shortest path"
	self.applyButtonDraw.connect('clicked(bool)', self.onApplyButtonDraw)
	
	#length
	self.lengthLineEdit = qt.QLineEdit()
	self.lengthLineEdit.text = '...'
	self.lengthLineEdit.readOnly = True
	self.lengthLineEdit.frame = True
	self.lengthLineEdit.styleSheet = "QLineEdit { background:transparent; }"
	self.lengthLineEdit.cursor = qt.QCursor(qt.Qt.IBeamCursor)
	parametersFormLayout.addRow("Length (cm):", self.lengthLineEdit)
	
	#####
	##10-20 system electrode distances area
	#####
	EEGCollapsibleButton = ctk.ctkCollapsibleButton()
	EEGCollapsibleButton.text = "10-20 system electrode"
	self.layout.addWidget(EEGCollapsibleButton)

	# Layout within the dummy collapsible button
	EEGFormLayout = qt.QFormLayout(EEGCollapsibleButton)

	# Source points (vtkMRMLMarkupsFiducialNode)
	self.SourceSelector2 = slicer.qMRMLNodeComboBox()
	self.SourceSelector2.nodeTypes = ( ("vtkMRMLMarkupsFiducialNode"), "" )
	self.SourceSelector2.addEnabled = True
	self.SourceSelector2.removeEnabled = False
	self.SourceSelector2.noneEnabled = True
	self.SourceSelector2.showHidden = False
	self.SourceSelector2.renameEnabled = True
	self.SourceSelector2.showChildNodeTypes = False
	self.SourceSelector2.setMRMLScene( slicer.mrmlScene )
	self.SourceSelector2.setToolTip( "Four anatomical landmarks are used for the essential positioning of the electrodes:1/the nasion 2/the inion 3/the pre auricular to the left ear 4/the pre auricular to the right ear" )
	EEGFormLayout.addRow("4 anatomical landmarks: ", self.SourceSelector2)

	# input model	
	self.inputTargetModelSelector = slicer.qMRMLNodeComboBox()
	self.inputTargetModelSelector.nodeTypes = ( ("vtkMRMLModelNode"), "" )
	self.inputTargetModelSelector.selectNodeUponCreation = True
	self.inputTargetModelSelector.addEnabled = False
	self.inputTargetModelSelector.removeEnabled = False
	self.inputTargetModelSelector.noneEnabled = False
	self.inputTargetModelSelector.showHidden = False
	self.inputTargetModelSelector.showChildNodeTypes = False
	self.inputTargetModelSelector.setMRMLScene( slicer.mrmlScene )
	self.inputTargetModelSelector.setToolTip( "Select the model the other will be transformed to. This model required to contain a dense set of points." )
	EEGFormLayout.addRow("Input STL model: ", self.inputTargetModelSelector)

	# Apply Button "Make 10-20 EEG system electrode"
	self.applyButton = qt.QPushButton("Make 10-20 EEG system electrode")
	self.applyButton.toolTip = "Make 10-20 EEG system electrode via the Dijkstra's algorithm"
	self.applyButton.enabled = True
	EEGFormLayout.addRow(self.applyButton)

	# connections "Make 10-20 EEG system electrode"
	self.applyButton.connect('clicked(bool)', self.onApplyEEG)
  
	# StimulationPoint
	self.stimulationSite = qt.QHBoxLayout()
	self.StimulationPointOff = qt.QRadioButton("No")
	self.StimulationPointOff.connect('clicked(bool)', self.onStimulationPointOff)
	self.StimulationPointOn = qt.QRadioButton("Yes")
	self.StimulationPointOn.connect('clicked(bool)', self.onStimulationPointOn)
	self.stimulationSite.addWidget(self.StimulationPointOff)
	self.stimulationSite.addWidget(self.StimulationPointOn)
	self.StimulationPointGroup = qt.QButtonGroup()
	self.StimulationPointGroup.addButton(self.StimulationPointOff)
	self.StimulationPointGroup.addButton(self.StimulationPointOn)
	self.StimulationPointOff.toolTip = "Is the stimulation Site with the Create-and-Place Ficudial button is placed?"
	self.StimulationPointOn.toolTip = "Is the stimulation Site with the Create-and-Place Ficudial button is placed?"

	EEGFormLayout.addRow("Stimulation Site placed:", self.stimulationSite)

	# Apply Button "Project the stimulation site"
	self.applyButtonProject = qt.QPushButton("Project the stimulation site")
	self.applyButtonProject.toolTip = "Project the stimulation site, the minimum euclidian distance between the scalp and the stimulation site, possible if the stimulation site placed"
	self.applyButtonProject.enabled = False
	EEGFormLayout.addRow(self.applyButtonProject)

	# connections "Project the stimulation site"
	self.applyButtonProject.connect('clicked(bool)', self.onApplyProject)
	
    # Nearest electrode data
	self.lengthElectrode1 = qt.QLineEdit()
	self.lengthElectrode1.text = '...'
	self.lengthElectrode1.readOnly = True
	self.lengthElectrode1.frame = True
	self.lengthElectrode1.styleSheet = "QLineEdit { background:transparent; }"
	self.lengthElectrode1.cursor = qt.QCursor(qt.Qt.IBeamCursor)
	self.lengthElectrode1.enabled = False    #True if the stimulation site placed
	EEGFormLayout.addRow("Nearest electrode 1:", self.lengthElectrode1)

	self.lengthElectrode2 = qt.QLineEdit()
	self.lengthElectrode2.text = '...'
	self.lengthElectrode2.readOnly = True
	self.lengthElectrode2.frame = True
	self.lengthElectrode2.styleSheet = "QLineEdit { background:transparent; }"
	self.lengthElectrode2.cursor = qt.QCursor(qt.Qt.IBeamCursor)
	self.lengthElectrode2.enabled = False
	EEGFormLayout.addRow("Nearest electrode 2:", self.lengthElectrode2)

	self.lengthElectrode3 = qt.QLineEdit()
	self.lengthElectrode3.text = '...'
	self.lengthElectrode3.readOnly = True
	self.lengthElectrode3.frame = True
	self.lengthElectrode3.styleSheet = "QLineEdit { background:transparent; }"
	self.lengthElectrode3.cursor = qt.QCursor(qt.Qt.IBeamCursor)
	self.lengthElectrode3.enabled = False
	EEGFormLayout.addRow("Nearest electrode 3:", self.lengthElectrode3)
	
	# Set default
	self.StimulationPointOff.setChecked(True)
	self.onStimulationPointOff(True)
	
	######
	##rTMS resting motor threshold-- Correction factor
	#####
	self.MTCollapsibleButton = ctk.ctkCollapsibleButton()
	self.MTCollapsibleButton.text = "rTMS resting motor threshold- Correction factor"
	self.layout.addWidget(self.MTCollapsibleButton)
	self.MTCollapsibleButton.collapsed = True

	# Layout within the dummy collapsible button
	MTFormLayout = qt.QFormLayout(self.MTCollapsibleButton)
	
	# M1 Point The primary motor cortex (Brodmann area 4)
	self.M1Site = qt.QHBoxLayout()
	self.M1SiteOff = qt.QRadioButton("No")
	self.M1SiteOff.connect('clicked(bool)', self.onM1SiteOff)
	self.M1SiteOn = qt.QRadioButton("Yes")
	self.M1SiteOn.connect('clicked(bool)', self.onM1SiteOn)
	self.M1Site.addWidget(self.M1SiteOff)
	self.M1Site.addWidget(self.M1SiteOn)
	self.M1SiteGroup = qt.QButtonGroup()
	self.M1SiteGroup.addButton(self.M1SiteOff)
	self.M1SiteGroup.addButton(self.M1SiteOn)
	self.M1SiteOff.toolTip = "Is the M1 site is placed?"
	self.M1SiteOn.toolTip = "Is the M1 site is placed?"

	MTFormLayout.addRow("M1 Point placed:", self.M1Site)
	
    # Value of the motor threshold (MT is the unadjusted MT in % stimulator output)
	self.MTunadjusted = ctk.ctkSliderWidget()
	self.MTunadjusted.singleStep = 1.0
	self.MTunadjusted.minimum = 0.0
	self.MTunadjusted.maximum = 150.0
	self.MTunadjusted.connect("valueChanged(double)", self.onTubeUpdated) #connection
	self.MTunadjusted.value = 100.0
	self.MTunadjusted.setToolTip("Value of the motor threshold (MT is the unadjusted MT in % stimulator output")
	self.MTunadjusted.enabled = False    #True if the stimulation site placed
	MTFormLayout.addRow("The unadjusted MT in % stimulator output: ", self.MTunadjusted)
	
	# Apply Button "Correct the  motor threshold"
	self.applyButtonCorrect = qt.QPushButton("Correct the motor threshold")
	self.applyButtonCorrect.toolTip = "Correct the  motor threshold"
	self.applyButtonCorrect.enabled = False
	MTFormLayout.addRow(self.applyButtonCorrect)

	# connections "Project the stimulation site"
	self.applyButtonCorrect.connect('clicked(bool)', self.onApplyCorrect)

    # Value of the motor threshold adjusted (Stokes et al. Clin  Neurophysiol 2007)
	self.MTadjusted = qt.QLineEdit()
	self.MTadjusted.text = '...'
	self.MTadjusted.readOnly = True
	self.MTadjusted.frame = True
	self.MTadjusted.styleSheet = "QLineEdit { background:transparent; }"
	self.MTadjusted.cursor = qt.QCursor(qt.Qt.IBeamCursor)
	self.MTadjusted.enabled = False    #True if the stimulation site placed
	MTFormLayout.addRow("The adjusted MT in % (according to Stokes, 2007):", self.MTadjusted) 
	
	# Value of the motor threshold adjusted (Hoffman et al. Biol Psychiatry 2013)
	self.MTadjusted2 = qt.QLineEdit()
	self.MTadjusted2.text = '...'
	self.MTadjusted2.readOnly = True
	self.MTadjusted2.frame = True
	self.MTadjusted2.styleSheet = "QLineEdit { background:transparent; }"
	self.MTadjusted2.cursor = qt.QCursor(qt.Qt.IBeamCursor)
	self.MTadjusted2.enabled = False    #True if the stimulation site placed
	MTFormLayout.addRow("The adjusted MT in % (according to Hoffman, 2013):", self.MTadjusted2) 
		
	# Set default
	self.M1SiteOff.setChecked(True)
	self.onM1SiteOff(True)
	
	# Add vertical spacer
	self.layout.addStretch(1)
	

  def cleanup(self):
    pass
	
  def onApplyButton2(self):
    logic = GeodesicSlicerLogic()
    self.applyButton2.text = "Working..."
    self.applyButton2.repaint()
    #self.applyButton2.repaint()
    slicer.app.processEvents()
    logic.mesh(self.inputSelector.currentNode()) 
    self.applyButton2.text = "Create a quick mesh"
  
  def onApplyButtonA(self):
    logic = GeodesicSlicerLogic()
    self.applyButtonA.text = "Working..."
    self.applyButtonA.repaint()
    #self.applyButtonA.repaint()
    slicer.app.processEvents()
    logic.mesh2(self.inputSelector.currentNode())
    self.applyButtonA.text = "Create a mesh"
	
  def onSelect(self):
    self.applyButton.enabled = self.inputTargetModelSelector.currentNode() and self.outputSelector.currentNode()

  def onApplyButton(self):
    logic = GeodesicSlicerLogic()
    slicer.app.processEvents()
    logic.run_djikstra(self.SourceSelector, self.lengthLineEdit, self.inputTargetModelSelector.currentNode()) 
	
  def onApplyButtonDraw(self):
    logic = GeodesicSlicerLogic()
    slicer.app.processEvents()
    logic.draw(self.SourceSelector, self.inputTargetModelSelector.currentNode()) 
	
  def onApplyEEG(self):
    logic = GeodesicSlicerLogic()
    slicer.app.processEvents()
    logic.EEG(self.SourceSelector2, self.inputTargetModelSelector.currentNode())
	
  def onStimulationPointOff(self, s):
    logic = GeodesicSlicerLogic()
    logic.setStimulationPoint(0)
    self.applyButtonProject.enabled = False
    self.lengthElectrode1.enabled = False
    self.lengthElectrode2.enabled = False
    self.lengthElectrode3.enabled = False
    self.lengthElectrode1.text = '...'
    self.lengthElectrode2.text = '...'
    self.lengthElectrode3.text = '...'
    
  def onStimulationPointOn(self, s):
    logic = GeodesicSlicerLogic()
    logic.setStimulationPoint(1)
    self.applyButtonProject.enabled = True
    self.lengthElectrode1.enabled = True
    self.lengthElectrode2.enabled = True
    self.lengthElectrode3.enabled = True
	
  def onApplyProject(self):
    logic = GeodesicSlicerLogic()
    slicer.app.processEvents()
    logic.ProjectedPoint(self.SourceSelector2, self.inputTargetModelSelector.currentNode(),self.lengthElectrode1, self.lengthElectrode2, self.lengthElectrode3)
	
  def onM1SiteOff(self, s):
	logic = GeodesicSlicerLogic()
	logic.setM1Site(0)
	self.MTunadjusted.enabled = False
	self.applyButtonProject.enabled = False
	self.MTadjusted.enabled = False
	self.MTadjusted2.enabled = False
	self.MTadjusted.text = '...'
    
  def onM1SiteOn(self, s):
	logic = GeodesicSlicerLogic()
	logic.setM1Site(1)
	self.applyButtonCorrect.enabled = True
	self.MTunadjusted.enabled = True
	self.MTadjusted.enabled = True
	self.MTadjusted2.enabled = True

  def onTubeUpdated(self, newValue):
	logic = GeodesicSlicerLogic()
	#print "MT unadjusted:", newValue
	logic.setMTunadjusted(newValue)
	self.skip = int(newValue)
	
  def onApplyCorrect(self):
    logic = GeodesicSlicerLogic()
    self.MTunadjusted = self.skip
    # print self.MTunadjusted
    slicer.app.processEvents()
    logic.CorrectedPoint(self.SourceSelector2, self.inputTargetModelSelector.currentNode(),self.MTunadjusted,self.MTadjusted,self.MTadjusted2)
	
#
# GeodesicSlicerLogic
#

class GeodesicSlicerLogic(ScriptedLoadableModuleLogic):

  def __init__(self):
	self.StimulationPoint = 0
	self.M1Site = 0
	self.MTunadjusted = 100
	
  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True

  def takeScreenshot(self,name,description,type=-1):
    # show the message even if not taking a screen shot
    slicer.util.delayDisplay('Take screenshot: '+description+'.\nResult is available in the Annotations module.', 3000)

    lm = slicer.app.layoutManager()
    # switch on the type to get the requested window
    widget = 0
    if type == slicer.qMRMLScreenShotDialog.FullLayout:
      # full layout
      widget = lm.viewport()
    elif type == slicer.qMRMLScreenShotDialog.ThreeD:
      # just the 3D window
      widget = lm.threeDWidget(0).threeDView()
    elif type == slicer.qMRMLScreenShotDialog.Red:
      # red slice window
      widget = lm.sliceWidget("Red")
    elif type == slicer.qMRMLScreenShotDialog.Yellow:
      # yellow slice window
      widget = lm.sliceWidget("Yellow")
    elif type == slicer.qMRMLScreenShotDialog.Green:
      # green slice window
      widget = lm.sliceWidget("Green")
    else:
      # default to using the full window
      widget = slicer.util.mainWindow()
      # reset the type so that the node is set correctly
      type = slicer.qMRMLScreenShotDialog.FullLayout

    # grab and convert to vtk image data
    qimage = ctk.ctkWidgetsUtils.grabWidget(widget)
    imageData = vtk.vtkImageData()
    slicer.qMRMLUtils().qImageToVtkImageData(qimage,imageData)

    annotationLogic = slicer.modules.annotations.logic()
    annotationLogic.CreateSnapShot(name, description, type, 1, imageData)

  def mesh(self, inputVolume):
	"""
	From the ExtractSkin.py of lassoan: https://gist.github.com/lassoan/1673b25d8e7913cbc245b4f09ed853f9
	"""
	# inputVolume added ?		
	if not inputVolume:
		error_text='Add a volume.nii'
		slicer.util.errorDisplay(error_text, windowTitle='Geodesic Slicer error', parent=None, standardButtons=None)
		return False
	
	logging.info('Processing started')
	# wait popup
	progressBar=slicer.util.createProgressDialog()
	progressBar.labelText='This can take a few minutes'
	slicer.app.processEvents()
	# model filename
	nom = inputVolume.GetName() 
	print nom
	masterVolumeNode = slicer.util.getNode(nom)
	##make the mesh
	# Create segmentation
	segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
	segmentationNode.CreateDefaultDisplayNodes() # only needed for display
	segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(masterVolumeNode)
	addedSegmentID = segmentationNode.GetSegmentation().AddEmptySegment("skin")
	# Create segment editor to get access to effects
	segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
	segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
	segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
	segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
	segmentEditorWidget.setSegmentationNode(segmentationNode)
	segmentEditorWidget.setMasterVolumeNode(masterVolumeNode)
	# Thresholding
	segmentEditorWidget.setActiveEffectByName("Threshold")
	effect = segmentEditorWidget.activeEffect()
	effect.setParameter("MinimumThreshold","100") #adjusting according to your data
	effect.setParameter("MaximumThreshold","5000") #adjusting according to your data
	effect.self().onApply()
	###display output
	progressBar.value = 10
	# Smoothing 1
	segmentEditorWidget.setActiveEffectByName("Smoothing")
	effect = segmentEditorWidget.activeEffect()
	effect.setParameter("SmoothingMethod", "MEDIAN")
	effect.setParameter("KernelSizeMm", 10)
	# effect.setParameter("SmoothingMethod", "GAUSSIAN")
	# effect.setParameter("GaussianStandardDeviationMm", 2)
	effect.self().onApply()
	###display output
	progressBar.value = 30
	# Clean up
	segmentEditorWidget = None
	slicer.mrmlScene.RemoveNode(segmentEditorNode)
	# Make segmentation results visible in 3D
	segmentationNode.CreateClosedSurfaceRepresentation()
	# Fix normals
	surfaceMesh = segmentationNode.GetClosedSurfaceRepresentation(addedSegmentID)
	normals = vtk.vtkPolyDataNormals() #to normal data
	normals.SetAutoOrientNormals(True) #to normal data
	normals.ConsistencyOn() #to normal data
	normals.SetInputData(surfaceMesh) #to normal data
	normals.Update() #to normal data
	surfaceMesh = normals.GetOutput() #to normal data
	###display output
	progressBar.value = 80
	#Write to STL file
	writer = vtk.vtkSTLWriter()
	writer.SetInputData(surfaceMesh)
	volumeNode2 = slicer.util.getNode(nom)
	filename2 = volumeNode2.GetStorageNode().GetFileName()
	nom=filename2+'.stl'
	logging.info('Model is writing')
	writer.SetFileName(nom)
	writer.Update()	
	##load this model
	name = inputVolume.GetName() 
	#print name
	volumeNode2 = slicer.util.getNode(name)
	filename2 = volumeNode2.GetStorageNode().GetFileName()
	name2=filename2+'.stl'
	#print name2
	slicer.util.loadModel((name2), returnNode=True)[1]
	progressBar.value = 100
	logging.info('Processing completed here '+nom)
	
  def mesh2(self,inputVolume):
	"""
	Since the ExtractSkin.py, with big smoothing effect (in order to projhect the stimulation site module)
	"""
	# inputVolume added ?		
	if not inputVolume:
		error_text='Add a volume.nii'
		slicer.util.errorDisplay(error_text, windowTitle='Geodesic Slicer error', parent=None, standardButtons=None)
		return False
	
	logging.info('Processing started')
	# wait popup
	progressBar=slicer.util.createProgressDialog()
	progressBar.labelText='This can take a few minutes'
	slicer.app.processEvents()
	# model filename
	nom = inputVolume.GetName() 
	print nom
	masterVolumeNode = slicer.util.getNode(nom)
	##make the mesh
	# Create segmentation
	segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
	segmentationNode.CreateDefaultDisplayNodes() # only needed for display
	segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(masterVolumeNode)
	addedSegmentID = segmentationNode.GetSegmentation().AddEmptySegment("skin")
	# Create segment editor to get access to effects
	segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
	segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
	segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
	segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
	segmentEditorWidget.setSegmentationNode(segmentationNode)
	segmentEditorWidget.setMasterVolumeNode(masterVolumeNode)
	# Thresholding
	segmentEditorWidget.setActiveEffectByName("Threshold")
	effect = segmentEditorWidget.activeEffect()
	effect.setParameter("MinimumThreshold","100") #adjusting according to your data
	effect.setParameter("MaximumThreshold","5000") #adjusting according to your data
	effect.self().onApply()
	###display output
	progressBar.value = 10
	time.sleep(5)
	# Smoothing 1
	segmentEditorWidget.setActiveEffectByName("Smoothing")
	effect = segmentEditorWidget.activeEffect()
	effect.setParameter("SmoothingMethod", "MEDIAN")
	effect.setParameter("KernelSizeMm", 10)
	effect.self().onApply()
	###display output
	progressBar.value = 30
	time.sleep(5)
	# Smoothing 2
	segmentEditorWidget.setActiveEffectByName("Smoothing")
	effect = segmentEditorWidget.activeEffect()
	effect.setParameter("SmoothingMethod", "MORPHOLOGICAL_CLOSING")
	effect.setParameter("KernelSizeMm", 20) #long time
	effect.self().onApply()
	# Clean up
	segmentEditorWidget = None
	slicer.mrmlScene.RemoveNode(segmentEditorNode)
	# Make segmentation results visible in 3D
	segmentationNode.CreateClosedSurfaceRepresentation()
	# Fix normals
	surfaceMesh = segmentationNode.GetClosedSurfaceRepresentation(addedSegmentID)
	normals = vtk.vtkPolyDataNormals() #to normal data
	normals.SetAutoOrientNormals(True) #to normal data
	normals.ConsistencyOn() #to normal data
	normals.SetInputData(surfaceMesh) #to normal data
	normals.Update() #to normal data
	surfaceMesh = normals.GetOutput() #to normal data
	###display output
	progressBar.value = 80
	time.sleep(3)
	#Write to STL file
	writer = vtk.vtkSTLWriter()
	writer.SetInputData(surfaceMesh)
	volumeNode2 = slicer.util.getNode(nom)
	filename2 = volumeNode2.GetStorageNode().GetFileName()
	nom=filename2+'.stl'
	logging.info('Model is writing')
	writer.SetFileName(nom)
	writer.Update()	
	##load this model
	name = inputVolume.GetName() 
	#print name
	volumeNode2 = slicer.util.getNode(name)
	filename2 = volumeNode2.GetStorageNode().GetFileName()
	name2=filename2+'.stl'
	#print name2
	slicer.util.loadModel((name2), returnNode=True)[1]
	progressBar.value = 100
	logging.info('Processing completed here '+nom)
	
  def run_djikstra(self, fiducialInput, lengthInput, inputModel):
	"""
	Run the Dijkstra's algorithm
	"""
	# fiducials created ?		
	if not fiducialInput.currentNode():
		error_text='Add at least two fiducials'
		slicer.util.errorDisplay(error_text, windowTitle='Geodesic Slicer error', parent=None, standardButtons=None)
		return False
	
	logging.info('Processing started')
	
	# text view
	view=slicer.app.layoutManager().threeDWidget(0).threeDView()
	view.cornerAnnotation().ClearAllTexts()
	view.forceRender()
	
	# model filename
	nom = inputModel.GetName() 
	volumeNode = slicer.util.getNode(nom)
	filename = volumeNode.GetStorageNode().GetFileName()
	logging.info('filename: ' + filename)
	
	# wait popup
	progressBar=slicer.util.createProgressDialog()
	#slicer.app.processEvents()
	
	# fiducial list
	numFids = fiducialInput.currentNode().GetNumberOfFiducials()
	
	list=[]
	for i in range(numFids):
		ras = [0,0,0]
		fiducialInput.currentNode().GetNthFiducialPosition(i,ras)
		#print i,": RAS =",ras
		list.append(ras)
	
	# locator	
	pd = inputModel.GetModelDisplayNode()
	pd1=pd.GetOutputPolyData()
	pd1.GetNumberOfPoints()
	loc = vtk.vtkPointLocator()
	loc.SetDataSet(pd1)
	loc.BuildLocator()
	closestPointId = loc.FindClosestPoint(list[0]) #fiducial 1
	closestPointId1 = loc.FindClosestPoint(list[1]) #fiducial 2
	
	fiducials = []
	for fiducial in list:
		fiducials.append(loc.FindClosestPoint(fiducial))
		
	##get the distance of the geodesic path 
	appendFilter = vtk.vtkAppendFilter()
	appendFilter.MergePointsOn()
	points = vtk.vtkPoints()
	
	p0 = [0,0,0]
	p1 = [0,0,0]
	dist = 0.0
	
	for n in range(len(fiducials)-1):
		v0 = fiducials[n]
		v1 = fiducials[n+1]
		
		#create geodesic path: vtkDijkstraGraphGeodesicPath
		dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
		dijkstra.SetInputConnection(pd.GetOutputPolyDataConnection())
		dijkstra.SetStartVertex(v0)
		dijkstra.SetEndVertex(v1)
		dijkstra.Update()
		
		pts = dijkstra.GetOutput().GetPoints()
		end = n<len(fiducials)-2 and 0 or -1
		for ptId in range(pts.GetNumberOfPoints()-1, end, -1):
			pts.GetPoint(ptId, p0)
			points.InsertNextPoint(p0)		

		for ptId in range(pts.GetNumberOfPoints()-1):
			pts.GetPoint(ptId, p0)
			pts.GetPoint(ptId+1, p1)
			dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))

		appendFilter.AddInputConnection(dijkstra.GetOutputPort())
								
	
	# display output
	print 'length= ' , (dist/10) , ' cm'
	lengthInput.text = (dist/10)
	
			
	# length text on model view
	view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperLeft,'Length = '+str(dist/10)+ ' cm') # Set text to "dist"
	view.cornerAnnotation().GetTextProperty().SetColor(1,0,0) # Set color to red
	view.forceRender()
		
	logging.info('Processing completed')
	
	progressBar.value = 100
	
  def draw(self, fiducialInput, inputModel):
	"""
	Draw the Dijkstra's algorithm
	"""
	# fiducials created ?		
	if not fiducialInput.currentNode():
		error_text='Add at least two fiducials'
		slicer.util.errorDisplay(error_text, windowTitle='Geodesic Slicer error', parent=None, standardButtons=None)
		return False
	
	logging.info('Processing started')
	
	# text view
	view=slicer.app.layoutManager().threeDWidget(0).threeDView()
	view.cornerAnnotation().ClearAllTexts()
	view.forceRender()
	
	# model filename
	nom = inputModel.GetName() 
	volumeNode = slicer.util.getNode(nom)
	filename = volumeNode.GetStorageNode().GetFileName()
	logging.info('filename: ' + filename)
	
	# wait popup
	progressBar=slicer.util.createProgressDialog()
	#slicer.app.processEvents()
	
	# fiducial list
	numFids = fiducialInput.currentNode().GetNumberOfFiducials()
	
	list=[]
	for i in range(numFids):
		ras = [0,0,0]
		fiducialInput.currentNode().GetNthFiducialPosition(i,ras)
		#print i,": RAS =",ras
		list.append(ras)
	
	# locator	
	pd = inputModel.GetModelDisplayNode()
	pd1=pd.GetOutputPolyData()
	pd1.GetNumberOfPoints()
	loc = vtk.vtkPointLocator()
	loc.SetDataSet(pd1)
	loc.BuildLocator()
	closestPointId = loc.FindClosestPoint(list[0]) #fiducial 1
	closestPointId1 = loc.FindClosestPoint(list[1]) #fiducial 2
	
	fiducials = []
	for fiducial in list:
		fiducials.append(loc.FindClosestPoint(fiducial))
		
	##get the distance of the geodesic path 
	appendFilter = vtk.vtkAppendFilter()
	appendFilter.MergePointsOn()
	points = vtk.vtkPoints()
	#vIds = [closestPointId,closestPointId1]
	p0 = [0,0,0]
	p1 = [0,0,0]
	dist = 0.0
	
	for n in range(len(fiducials)-1):
		v0 = fiducials[n]
		v1 = fiducials[n+1]
		
		m = n +1
		#print m
		if m > 0:
			progressBar.value = 100-(100/m)
			slicer.app.processEvents()
		
		model2 = slicer.util.loadModel(filename, returnNode=True)[1]
		model2.GetDisplayNode().SetColor(1,0,0)
		dijkstra2 = vtk.vtkDijkstraGraphGeodesicPath()
		dijkstra2.SetInputConnection(model2.GetPolyDataConnection())
		model2.SetPolyDataConnection(dijkstra2.GetOutputPort())
		dijkstra2.SetStartVertex(v0) #start
		dijkstra2.SetEndVertex(v1) #end
		dijkstra2.Update()
		
	logging.info('Processing completed')
	progressBar.value = 100

	return True
	
  def EEG(self, fiducialInput, inputModel):
	"""
	Run the Dijkstra's algorithm to make the 10-20 system electrode distances
	"""
			# fiducials created ?		
	if not fiducialInput.currentNode():
		error_text='Add 4 anatomical landmarks'
		slicer.util.errorDisplay(error_text, windowTitle='Geodesic Slicer error', parent=None, standardButtons=None)
		#progressBar.close()
		return False
	
	# fiducial list
	numFids = fiducialInput.currentNode().GetNumberOfFiducials()
	
	# fiducials created ?			
	if  numFids!=4:
		error_text='Add 4 anatomical landmarks'
		slicer.util.errorDisplay(error_text, windowTitle='Geodesic Slicer error', parent=None, standardButtons=None)
		#progressBar.close()
		return False
	
	logging.info('Processing started')
	# # wait popup
	progressBar=slicer.util.createProgressDialog()	
	
	# model filename
	nom = inputModel.GetName() 
	volumeNode = slicer.util.getNode(nom)
	filename = volumeNode.GetStorageNode().GetFileName()
	logging.info('filename: ' + filename)
		
	list=[]
	for i in range(numFids):
	  ras = [0,0,0]
	  fiducialInput.currentNode().GetNthFiducialPosition(i,ras)
	  #print i,": RAS =",ras
	  list.append(ras)
	  
	#create point (Near_Cz) de passage au niveau du top of the scalp
	nasion=list[0]
	inion=list[1]
	A1=list[2] #left tragus
	A2=list[3] #right tragus
	b=[0]*6 #(xmin,xmax, ymin,ymax, zmin,zmax)
	
	inputModel.GetPolyData().GetBounds(b)
	x=[list[1][0],list[0][0]]
	y=[list[2][2],list[3][2]]
	Near_Cz=[mean(x),mean(y),b[5]]
	
	fidNode = fiducialInput.currentNode() #pour changement nom

	#locator
	pd = inputModel.GetModelDisplayNode()
	pd1=pd.GetOutputPolyData()
	pd1.GetNumberOfPoints()
	loc = vtk.vtkPointLocator()
	loc.SetDataSet(pd1)
	loc.BuildLocator()

	closestPointId = loc.FindClosestPoint(nasion) #Nasion 
	closestPointId1 = loc.FindClosestPoint(inion) #Inion
	closestPointId2 = loc.FindClosestPoint(A1) #left tragus
	closestPointId3 = loc.FindClosestPoint(A2) #right tragus
	closestPointIdCz_tmp_cor = loc.FindClosestPoint(Near_Cz) #to determine Cz
	closestPointIdCz_tmp_sag=0 #to determine Cz
	
	while closestPointIdCz_tmp_cor !=closestPointIdCz_tmp_sag:
		##get the distance of the geodesic path 
		appendFilter = vtk.vtkAppendFilter()
		appendFilter.MergePointsOn()
		points = vtk.vtkPoints()
		#//begin Cz Ligne EEG Sagital
		vIds = [closestPointId,closestPointIdCz_tmp_cor,closestPointId1]
		p0 = [0,0,0]
		p1 = [0,0,0]
		dist = 0.0
		for n in range(len(vIds)-1):
			v0 = vIds[n]
			v1 = vIds[n+1]
			
			#create geodesic path: vtkDijkstraGraphGeodesicPath
			dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
			dijkstra.SetInputConnection(pd.GetOutputPolyDataConnection())
			dijkstra.SetStartVertex(v0)
			dijkstra.SetEndVertex(v1)
			dijkstra.Update()
						
			pts = dijkstra.GetOutput().GetPoints()
			end = n<len(vIds)-2 and 0 or -1
			for ptId in range(pts.GetNumberOfPoints()-1, end, -1):
				pts.GetPoint(ptId, p0)
				points.InsertNextPoint(p0)		
			
			for ptId in range(pts.GetNumberOfPoints()-1):
				pts.GetPoint(ptId, p0)
				pts.GetPoint(ptId+1, p1)
				dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
				
			appendFilter.AddInputConnection(dijkstra.GetOutputPort())
		Length_sagital2=dist/10
		appendFilter.Update()
		xSpline = vtk.vtkSCurveSpline()
		ySpline = vtk.vtkSCurveSpline()
		zSpline = vtk.vtkSCurveSpline()
		spline = vtk.vtkParametricSpline()
		spline.ParameterizeByLengthOn()
		spline.SetXSpline(xSpline)
		spline.SetYSpline(ySpline)
		spline.SetZSpline(zSpline)
		spline.SetPoints(points)
		functionSource = vtk.vtkParametricFunctionSource()
		functionSource.SetParametricFunction(spline)
		functionSource.Update()
		#Cz_tmp_sag
		u = [0.5,0,0]
		Cz_tmp_sag = [0]*3
		du = [0]*9
		spline.Evaluate(u, Cz_tmp_sag, du)
		closestPointIdCz_tmp_sag = loc.FindClosestPoint(Cz_tmp_sag)
		#print 'Cz_tmp_sag= ', Cz_tmp_sag, closestPointIdCz_tmp_sag
		#//begin Cz Ligne EEG Coronal
		vIds2 = [closestPointId3,closestPointIdCz_tmp_sag,closestPointId2]
		##get the distance of the geodesic path 
		points2 = vtk.vtkPoints()
		p0 = [0,0,0]
		p1 = [0,0,0]
		dist2 = 0.0
		for n in range(len(vIds2)-1):
			v0 = vIds2[n]
			v1 = vIds2[n+1]
			
			#create geodesic path: vtkDijkstraGraphGeodesicPath
			dijkstra2 = vtk.vtkDijkstraGraphGeodesicPath()
			dijkstra2.SetInputConnection(pd.GetOutputPolyDataConnection())
			dijkstra2.SetStartVertex(v0)
			dijkstra2.SetEndVertex(v1)
			dijkstra2.Update()
						
			pts = dijkstra2.GetOutput().GetPoints()
			end = n<len(vIds2)-2 and 0 or -1
			for ptId in range(pts.GetNumberOfPoints()-1, end, -1):
				pts.GetPoint(ptId, p0)
				points2.InsertNextPoint(p0)		
			
			for ptId in range(pts.GetNumberOfPoints()-1):
				pts.GetPoint(ptId, p0)
				pts.GetPoint(ptId+1, p1)
				dist2 += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
				
			appendFilter.AddInputConnection(dijkstra2.GetOutputPort())
		Length_coronal2=dist2/10
		appendFilter.Update()
		xSpline2 = vtk.vtkSCurveSpline()
		ySpline2 = vtk.vtkSCurveSpline()
		zSpline2 = vtk.vtkSCurveSpline()
		spline2 = vtk.vtkParametricSpline()
		spline2.ParameterizeByLengthOn()
		spline2.SetXSpline(xSpline2)
		spline2.SetYSpline(ySpline2)
		spline2.SetZSpline(zSpline2)
		spline2.SetPoints(points2)
		functionSource2 = vtk.vtkParametricFunctionSource()
		functionSource2.SetParametricFunction(spline2)
		functionSource2.Update()
		#Cz_tmp_cor
		u = [0.5,0,0]
		Cz_tmp_cor = [0]*3
		du = [0]*9
		spline2.Evaluate(u, Cz_tmp_cor, du)
		closestPointIdCz_tmp_cor = loc.FindClosestPoint(Cz_tmp_cor)
		#print 'Cz_tmp_cor= ', Cz_tmp_cor, closestPointIdCz_tmp_cor
		if closestPointIdCz_tmp_cor == closestPointIdCz_tmp_sag:
			print 'cor= ', Length_coronal2, 'sag= ',Length_sagital2
			num_Cz_tmp_cor=slicer.modules.markups.logic().AddFiducial(Cz_tmp_cor[0],Cz_tmp_cor[1],Cz_tmp_cor[2])
			fidNode.SetNthFiducialLabel(num_Cz_tmp_cor, "Cz")
			Cz=Cz_tmp_cor
			closestPointIdCz=closestPointIdCz_tmp_cor
			break
	#Ligne EEG sagittal
	##get the distance of the geodesic path 
	appendFilter = vtk.vtkAppendFilter()
	appendFilter.MergePointsOn()
	points = vtk.vtkPoints()
	p0 = [0,0,0]
	p1 = [0,0,0]
	dist = 0.0
	#begin Ligne EEG Sagital
	vIds = [closestPointId,closestPointIdCz,closestPointId1]
	p0 = [0,0,0]
	p1 = [0,0,0]
	dist = 0.0
	for n in range(len(vIds)-1):
		v0 = vIds[n]
		v1 = vIds[n+1]
		#create geodesic path: vtkDijkstraGraphGeodesicPath
		dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
		dijkstra.SetInputConnection(pd.GetOutputPolyDataConnection())
		dijkstra.SetStartVertex(v0)
		dijkstra.SetEndVertex(v1)
		dijkstra.Update()
		pts = dijkstra.GetOutput().GetPoints()
		end = n<len(vIds)-2 and 0 or -1
		for ptId in range(pts.GetNumberOfPoints()-1, end, -1):
			pts.GetPoint(ptId, p0)
			points.InsertNextPoint(p0)		
		for ptId in range(pts.GetNumberOfPoints()-1):
			pts.GetPoint(ptId, p0)
			pts.GetPoint(ptId+1, p1)
			dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
		appendFilter.AddInputConnection(dijkstra.GetOutputPort())
	Length_sagital=dist/10
	appendFilter.Update()
	xSpline = vtk.vtkSCurveSpline()
	ySpline = vtk.vtkSCurveSpline()
	zSpline = vtk.vtkSCurveSpline()
	spline = vtk.vtkParametricSpline()
	spline.ParameterizeByLengthOn()
	spline.SetXSpline(xSpline)
	spline.SetYSpline(ySpline)
	spline.SetZSpline(zSpline)
	spline.SetPoints(points)
	functionSource = vtk.vtkParametricFunctionSource()
	functionSource.SetParametricFunction(spline)
	functionSource.Update()
	#Fpz
	u = [0.1,0,0]
	Fpz = [0]*3
	du = [0]*9
	spline.Evaluate(u, Fpz, du)
	#print 'Fpz= ', Fpz
	num_Fpz=slicer.modules.markups.logic().AddFiducial(Fpz[0],Fpz[1],Fpz[2])
	fidNode.SetNthFiducialLabel(num_Fpz, "Fpz") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	#Fz
	u = [0.3,0,0]
	Fz = [0]*3
	du = [0]*9
	spline.Evaluate(u, Fz, du)
	#print 'Fz= ', Fz
	num_Fz=slicer.modules.markups.logic().AddFiducial(Fz[0],Fz[1],Fz[2])
	fidNode.SetNthFiducialLabel(num_Fz, "Fz") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	#Pz
	u = [0.7,0,0]
	Pz = [0]*3
	du = [0]*9
	spline.Evaluate(u, Pz, du)
	#print 'Pz= ', Pz
	num_Pz=slicer.modules.markups.logic().AddFiducial(Pz[0],Pz[1],Pz[2])
	fidNode.SetNthFiducialLabel(num_Pz, "Pz") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	#Oz
	u = [0.9,0,0]
	Oz = [0]*3
	du = [0]*9
	spline.Evaluate(u, Oz, du)
	#print 'Oz= ', Oz
	num_Oz=slicer.modules.markups.logic().AddFiducial(Oz[0],Oz[1],Oz[2])
	fidNode.SetNthFiducialLabel(num_Oz, "Oz") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	# ###display output
	progressBar.value = 14
	#Ligne EEG Coronal
	vIds2 = [closestPointId2,closestPointIdCz,closestPointId3]
	#get the distance of the geodesic path 
	points2 = vtk.vtkPoints()
	p0 = [0,0,0]
	p1 = [0,0,0]
	dist = 0.0
	for n in range(len(vIds2)-1):
		v0 = vIds2[n]
		v1 = vIds2[n+1]
		#create geodesic path: vtkDijkstraGraphGeodesicPath
		dijkstra2 = vtk.vtkDijkstraGraphGeodesicPath()
		dijkstra2.SetInputConnection(pd.GetOutputPolyDataConnection())
		dijkstra2.SetStartVertex(v0)
		dijkstra2.SetEndVertex(v1)
		dijkstra2.Update()
		pts = dijkstra2.GetOutput().GetPoints()
		end = n<len(vIds2)-2 and 0 or -1
		for ptId in range(pts.GetNumberOfPoints()-1, end, -1):
			pts.GetPoint(ptId, p0)
			points2.InsertNextPoint(p0)		
		for ptId in range(pts.GetNumberOfPoints()-1):
			pts.GetPoint(ptId, p0)
			pts.GetPoint(ptId+1, p1)
			dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
		appendFilter.AddInputConnection(dijkstra2.GetOutputPort())
	appendFilter.Update()
	xSpline2 = vtk.vtkSCurveSpline()
	ySpline2 = vtk.vtkSCurveSpline()
	zSpline2 = vtk.vtkSCurveSpline()
	spline2 = vtk.vtkParametricSpline()
	spline2.ParameterizeByLengthOn()
	spline2.SetXSpline(xSpline2)
	spline2.SetYSpline(ySpline2)
	spline2.SetZSpline(zSpline2)
	spline2.SetPoints(points2)
	functionSource2 = vtk.vtkParametricFunctionSource()
	functionSource2.SetParametricFunction(spline2)
	functionSource2.Update()
	#T3
	u = [0.1,0,0]
	T3 = [0]*3
	du = [0]*9
	spline2.Evaluate(u, T3, du)
	#print 'T3= ', T3
	num_T3=slicer.modules.markups.logic().AddFiducial(T3[0],T3[1],T3[2])
	fidNode.SetNthFiducialLabel(num_T3, "T3") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	#C3
	u = [0.3,0,0]
	C3 = [0]*3
	du = [0]*9
	spline2.Evaluate(u, C3, du)
	#print 'C3= ', C3
	num_C3=slicer.modules.markups.logic().AddFiducial(C3[0],C3[1],C3[2])
	fidNode.SetNthFiducialLabel(num_C3, "C3") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	closestPointIdC3 = loc.FindClosestPoint(C3)
	#C4
	u = [0.7,0,0]
	C4 = [0]*3
	du = [0]*9
	spline2.Evaluate(u, C4, du)
	#print 'C4= ', C4
	num_C4=slicer.modules.markups.logic().AddFiducial(C4[0],C4[1],C4[2])
	fidNode.SetNthFiducialLabel(num_C4, "C4") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	closestPointIdC4 = loc.FindClosestPoint(C4)
	#T4
	u = [0.9,0,0]
	T4 = [0]*3
	du = [0]*9
	spline2.Evaluate(u, T4, du)
	#print 'T4= ', T4
	num_T4=slicer.modules.markups.logic().AddFiducial(T4[0],T4[1],T4[2])
	fidNode.SetNthFiducialLabel(num_T4, "T4") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	# ###display output
	progressBar.value = 28
	#Ligne EEG Left
	closestPointIdOz = loc.FindClosestPoint(Oz)
	closestPointIdT3 = loc.FindClosestPoint(T3)
	closestPointIdFpz = loc.FindClosestPoint(Fpz)
	vIds3 = [closestPointIdOz,closestPointIdT3,closestPointIdFpz]
	##get the distance of the geodesic path 
	points3 = vtk.vtkPoints()
	p0 = [0,0,0]
	p1 = [0,0,0]
	dist = 0.0
	for n in range(len(vIds3)-1):
		v0 = vIds3[n]
		v1 = vIds3[n+1]
		#create geodesic path: vtkDijkstraGraphGeodesicPath
		dijkstra3 = vtk.vtkDijkstraGraphGeodesicPath()
		dijkstra3.SetInputConnection(pd.GetOutputPolyDataConnection())
		dijkstra3.SetStartVertex(v0)
		dijkstra3.SetEndVertex(v1)
		dijkstra3.Update()
		pts = dijkstra3.GetOutput().GetPoints()
		end = n<len(vIds3)-2 and 0 or -1
		for ptId in range(pts.GetNumberOfPoints()-1, end, -1):
			pts.GetPoint(ptId, p0)
			points3.InsertNextPoint(p0)		
		for ptId in range(pts.GetNumberOfPoints()-1):
			pts.GetPoint(ptId, p0)
			pts.GetPoint(ptId+1, p1)
			dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
		appendFilter.AddInputConnection(dijkstra3.GetOutputPort())
	appendFilter.Update()
	xSpline3 = vtk.vtkSCurveSpline()
	ySpline3 = vtk.vtkSCurveSpline()
	zSpline3 = vtk.vtkSCurveSpline() 
	spline3 = vtk.vtkParametricSpline()
	spline3.ParameterizeByLengthOn()
	spline3.SetXSpline(xSpline3)
	spline3.SetYSpline(ySpline3)
	spline3.SetZSpline(zSpline3)
	spline3.SetPoints(points3)
	functionSource3 = vtk.vtkParametricFunctionSource()
	functionSource3.SetParametricFunction(spline3)
	functionSource3.Update()
	#O1 ###
	u = [0.1,0,0]
	O1 = [0]*3
	du = [0]*9
	spline3.Evaluate(u, O1, du)
	#print 'O1= ', O1
	num_O1=slicer.modules.markups.logic().AddFiducial(O1[0],O1[1],O1[2])
	fidNode.SetNthFiducialLabel(num_O1, "O1") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	closestPointIdO1 = loc.FindClosestPoint(O1)
	#T5
	u = [0.3,0,0]
	T5 = [0]*3
	du = [0]*9
	spline3.Evaluate(u, T5, du)
	#print 'T5= ', T5
	num_T5=slicer.modules.markups.logic().AddFiducial(T5[0],T5[1],T5[2])
	fidNode.SetNthFiducialLabel(num_T5, "T5") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	closestPointIdT5 = loc.FindClosestPoint(T5)
	#F7
	u = [0.7,0,0]
	F7 = [0]*3
	du = [0]*9
	spline3.Evaluate(u, F7, du)
	#print 'F7= ', F7
	num_F7=slicer.modules.markups.logic().AddFiducial(F7[0],F7[1],F7[2])
	fidNode.SetNthFiducialLabel(num_F7, "F7") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	#Fp1
	u = [0.9,0,0]
	Fp1 = [0]*3
	du = [0]*9
	spline3.Evaluate(u, Fp1, du)
	#print 'Fp1= ', Fp1
	num_Fp1=slicer.modules.markups.logic().AddFiducial(Fp1[0],Fp1[1],Fp1[2])
	fidNode.SetNthFiducialLabel(num_Fp1, "Fp1") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	closestPointIdFp1 = loc.FindClosestPoint(Fp1)	
	# ###display output
	progressBar.value = 42
	#Ligne EEG right
	closestPointIdOz = loc.FindClosestPoint(Oz)
	closestPointIdT4 = loc.FindClosestPoint(T4)
	closestPointIdFpz = loc.FindClosestPoint(Fpz)
	vIds4 = [closestPointIdOz,closestPointIdT4,closestPointIdFpz]
	##get the distance of the geodesic path
	points4 = vtk.vtkPoints()
	p0 = [0,0,0]
	p1 = [0,0,0]
	dist = 0.0
	for n in range(len(vIds4)-1):
		v0 = vIds4[n]
		v1 = vIds4[n+1]
		#create geodesic path: vtkDijkstraGraphGeodesicPath
		dijkstra4 = vtk.vtkDijkstraGraphGeodesicPath()
		dijkstra4.SetInputConnection(pd.GetOutputPolyDataConnection())
		dijkstra4.SetStartVertex(v0)
		dijkstra4.SetEndVertex(v1)
		dijkstra4.Update()
		pts = dijkstra4.GetOutput().GetPoints()
		end = n<len(vIds4)-2 and 0 or -1
		for ptId in range(pts.GetNumberOfPoints()-1, end, -1):
			pts.GetPoint(ptId, p0)
			points4.InsertNextPoint(p0)		
		for ptId in range(pts.GetNumberOfPoints()-1):
			pts.GetPoint(ptId, p0)
			pts.GetPoint(ptId+1, p1)
			dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
		appendFilter.AddInputConnection(dijkstra4.GetOutputPort())
	appendFilter.Update()
	xSpline4 = vtk.vtkSCurveSpline()
	ySpline4 = vtk.vtkSCurveSpline()
	zSpline4 = vtk.vtkSCurveSpline()
	spline4 = vtk.vtkParametricSpline()
	spline4.ParameterizeByLengthOn()
	spline4.SetXSpline(xSpline4)
	spline4.SetYSpline(ySpline4)
	spline4.SetZSpline(zSpline4)
	spline4.SetPoints(points4)
	functionSource4 = vtk.vtkParametricFunctionSource()
	functionSource4.SetParametricFunction(spline4)
	functionSource4.Update()
	#O2 ###
	u = [0.1,0,0]
	O2 = [0]*3
	du = [0]*9
	spline4.Evaluate(u, O2, du)
	#print 'O2= ', O2
	num_O2=slicer.modules.markups.logic().AddFiducial(O2[0],O2[1],O2[2])
	fidNode.SetNthFiducialLabel(num_O2, "O2") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	closestPointIdO2 = loc.FindClosestPoint(O2)
	#T6
	u = [0.3,0,0]
	T6 = [0]*3
	du = [0]*9
	spline4.Evaluate(u, T6, du)
	#print 'T6= ', T6
	num_T6=slicer.modules.markups.logic().AddFiducial(T6[0],T6[1],T6[2])
	fidNode.SetNthFiducialLabel(num_T6, "T6") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	#F8
	u = [0.7,0,0]
	F8 = [0]*3
	du = [0]*9
	spline4.Evaluate(u, F8, du)
	#print 'F8= ', F8
	num_F8=slicer.modules.markups.logic().AddFiducial(F8[0],F8[1],F8[2])
	fidNode.SetNthFiducialLabel(num_F8, "F8") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	#Fp2
	u = [0.9,0,0]
	Fp2 = [0]*3
	du = [0]*9
	spline4.Evaluate(u, Fp2, du)
	#print 'Fp2= ', Fp2
	num_Fp2=slicer.modules.markups.logic().AddFiducial(Fp2[0],Fp2[1],Fp2[2])
	fidNode.SetNthFiducialLabel(num_Fp2, "Fp2") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	closestPointIdFp2 = loc.FindClosestPoint(Fp2)	
	# ###display output
	progressBar.value = 56
	#Ligne EEG Coronal bas
	closestPointIdT5 = loc.FindClosestPoint(T5)
	closestPointIdPz = loc.FindClosestPoint(Pz)
	closestPointIdT6 = loc.FindClosestPoint(T6)
	vIds5 = [closestPointIdT5,closestPointIdPz,closestPointIdT6]
	##get the distance of the geodesic path
	points5 = vtk.vtkPoints()
	p0 = [0,0,0]
	p1 = [0,0,0]
	dist = 0.0
	for n in range(len(vIds5)-1):
		v0 = vIds5[n]
		v1 = vIds5[n+1]
		#create geodesic path: vtkDijkstraGraphGeodesicPath
		dijkstra5 = vtk.vtkDijkstraGraphGeodesicPath()
		dijkstra5.SetInputConnection(pd.GetOutputPolyDataConnection())
		dijkstra5.SetStartVertex(v0)
		dijkstra5.SetEndVertex(v1)
		dijkstra5.Update()
		pts = dijkstra5.GetOutput().GetPoints()
		end = n<len(vIds5)-2 and 0 or -1
		for ptId in range(pts.GetNumberOfPoints()-1, end, -1):
			pts.GetPoint(ptId, p0)
			points5.InsertNextPoint(p0)		
		for ptId in range(pts.GetNumberOfPoints()-1):
			pts.GetPoint(ptId, p0)
			pts.GetPoint(ptId+1, p1)
			dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
		appendFilter.AddInputConnection(dijkstra5.GetOutputPort())
	appendFilter.Update()
	xSpline5 = vtk.vtkSCurveSpline()
	ySpline5 = vtk.vtkSCurveSpline()
	zSpline5 = vtk.vtkSCurveSpline()
	spline5 = vtk.vtkParametricSpline()
	spline5.ParameterizeByLengthOn()
	spline5.SetXSpline(xSpline5)
	spline5.SetYSpline(ySpline5)
	spline5.SetZSpline(zSpline5)
	spline5.SetPoints(points5)
	functionSource5 = vtk.vtkParametricFunctionSource()
	functionSource5.SetParametricFunction(spline5)
	functionSource5.Update()
	#P3 ###
	u = [0.25,0,0]
	P3 = [0]*3
	du = [0]*9
	spline5.Evaluate(u, P3, du)
	#print 'P3= ', P3
	num_P3=slicer.modules.markups.logic().AddFiducial(P3[0],P3[1],P3[2])
	fidNode.SetNthFiducialLabel(num_P3, "P3") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	#P4
	u = [0.75,0,0]
	P4 = [0]*3
	du = [0]*9
	spline5.Evaluate(u, P4, du)
	#print 'P4= ', P4
	num_P4=slicer.modules.markups.logic().AddFiducial(P4[0],P4[1],P4[2])
	fidNode.SetNthFiducialLabel(num_P4, "P4") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	closestPointIdP4 = loc.FindClosestPoint(P4)
	# ###display output
	progressBar.value = 70
	#Ligne EEG Coronal haut
	closestPointIdF7 = loc.FindClosestPoint(F7)
	closestPointIdFz = loc.FindClosestPoint(Fz)
	closestPointIdF8 = loc.FindClosestPoint(F8)
	vIds6 = [closestPointIdF7,closestPointIdFz,closestPointIdF8]
	##get the distance of the geodesic path
	points6 = vtk.vtkPoints()
	p0 = [0,0,0]
	p1 = [0,0,0]
	dist = 0.0
	for n in range(len(vIds6)-1):
		v0 = vIds6[n]
		v1 = vIds6[n+1]
		#create geodesic path: vtkDijkstraGraphGeodesicPath
		dijkstra6 = vtk.vtkDijkstraGraphGeodesicPath()
		dijkstra6.SetInputConnection(pd.GetOutputPolyDataConnection())
		dijkstra6.SetStartVertex(v0)
		dijkstra6.SetEndVertex(v1)
		dijkstra6.Update()
		pts = dijkstra6.GetOutput().GetPoints()
		end = n<len(vIds6)-2 and 0 or -1
		for ptId in range(pts.GetNumberOfPoints()-1, end, -1):
			pts.GetPoint(ptId, p0)
			points6.InsertNextPoint(p0)		
		for ptId in range(pts.GetNumberOfPoints()-1):
			pts.GetPoint(ptId, p0)
			pts.GetPoint(ptId+1, p1)
			dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
		appendFilter.AddInputConnection(dijkstra6.GetOutputPort())
	appendFilter.Update()
	xSpline6 = vtk.vtkSCurveSpline()
	ySpline6 = vtk.vtkSCurveSpline()
	zSpline6 = vtk.vtkSCurveSpline()
	spline6 = vtk.vtkParametricSpline()
	spline6.ParameterizeByLengthOn()
	spline6.SetXSpline(xSpline6)
	spline6.SetYSpline(ySpline6)
	spline6.SetZSpline(zSpline6)
	spline6.SetPoints(points6)
	functionSource6 = vtk.vtkParametricFunctionSource()
	functionSource6.SetParametricFunction(spline6)
	functionSource6.Update()
	#F3 ###
	u = [0.25,0,0]
	F3 = [0]*3
	du = [0]*9
	spline6.Evaluate(u, F3, du)
	#print 'F3= ', F3
	num_F3=slicer.modules.markups.logic().AddFiducial(F3[0],F3[1],F3[2])
	fidNode.SetNthFiducialLabel(num_F3, "F3") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	closestPointIdF3 = loc.FindClosestPoint(F3)
	#F4
	u = [0.75,0,0]
	F4 = [0]*3
	du = [0]*9
	spline6.Evaluate(u, F4, du)
	#print 'F4= ', F4
	num_F4=slicer.modules.markups.logic().AddFiducial(F4[0],F4[1],F4[2])
	fidNode.SetNthFiducialLabel(num_F4, "F4") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	closestPointIdF4 = loc.FindClosestPoint(F4)
	# ###display output
	progressBar.value = 84
	#//begin T3_P3 zone de stim
	closestPointIdT3 = loc.FindClosestPoint(T3)
	closestPointIdP3 = loc.FindClosestPoint(P3)
	vIds7 = [closestPointIdT3,closestPointIdP3]
	##get the distance of the geodesic path
	points7 = vtk.vtkPoints()
	p0 = [0,0,0]
	p1 = [0,0,0]
	dist = 0.0
	for n in range(len(vIds7)-1):
		v0 = vIds7[n]
		v1 = vIds7[n+1]
		#create geodesic path: vtkDijkstraGraphGeodesicPath
		dijkstra7 = vtk.vtkDijkstraGraphGeodesicPath()
		dijkstra7.SetInputConnection(pd.GetOutputPolyDataConnection())
		dijkstra7.SetStartVertex(v0)
		dijkstra7.SetEndVertex(v1)
		dijkstra7.Update()
		pts = dijkstra7.GetOutput().GetPoints()
		end = n<len(vIds7)-2 and 0 or -1
		for ptId in range(pts.GetNumberOfPoints()-1, end, -1):
			pts.GetPoint(ptId, p0)
			points7.InsertNextPoint(p0)		
		for ptId in range(pts.GetNumberOfPoints()-1):
			pts.GetPoint(ptId, p0)
			pts.GetPoint(ptId+1, p1)
			dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
		appendFilter.AddInputConnection(dijkstra7.GetOutputPort())
	Length_T3P3=dist/10
	appendFilter.Update()
	xSpline7 = vtk.vtkSCurveSpline()
	ySpline7 = vtk.vtkSCurveSpline()
	zSpline7 = vtk.vtkSCurveSpline()
	spline7 = vtk.vtkParametricSpline()
	spline7.ParameterizeByLengthOn()
	spline7.SetXSpline(xSpline7)
	spline7.SetYSpline(ySpline7)
	spline7.SetZSpline(zSpline7)
	spline7.SetPoints(points7)
	functionSource7 = vtk.vtkParametricFunctionSource()
	functionSource7.SetParametricFunction(spline7)
	functionSource7.Update()
	#T3P3 ###
	u = [0.5,0,0]
	T3P3 = [0]*3
	du = [0]*9
	spline7.Evaluate(u, T3P3, du)
	num_T3P3=slicer.modules.markups.logic().AddFiducial(T3P3[0],T3P3[1],T3P3[2])
	fidNode.SetNthFiducialLabel(num_T3P3, "T3P3") #chgt nom; n correspond au points 0 nazion ,1 inion,...
	# ###display output
	progressBar.value = 90	
	#Control process sagital
	vIds_sag1 = [closestPointIdCz,closestPointId]
	vIds_sag2 = [closestPointIdCz,closestPointId1]
	vIds_cor1 = [closestPointIdCz,closestPointId2]
	vIds_cor2 = [closestPointIdCz,closestPointId3]
	list_vIds_ctrl=[vIds_sag1,vIds_sag2,vIds_cor1,vIds_cor2]
	#get the distance of the geodesic path 
	dist_geodesic_ctrl_procs=[]
	for alpha in range(len(list_vIds_ctrl)):
		vIds_ctrl=list_vIds_ctrl[alpha]
		p0 = [0,0,0]
		p1 = [0,0,0]
		dist = 0.0
		#print vIds_ctrl
		for n in range(len(vIds_ctrl)-1):
			v0 = vIds_ctrl[n]
			v1 = vIds_ctrl[n+1]
			#create geodesic path: vtkDijkstraGraphGeodesicPath
			dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
			dijkstra.SetInputConnection(pd.GetOutputPolyDataConnection())
			dijkstra.SetStartVertex(v0)
			dijkstra.SetEndVertex(v1)
			dijkstra.Update()
			pts = dijkstra.GetOutput().GetPoints()
			end = n<len(vIds_ctrl)-2 and 0 or -1
			for ptId in range(pts.GetNumberOfPoints()-1, end, -1):
				pts.GetPoint(ptId, p0)
				points.InsertNextPoint(p0)		
			for ptId in range(pts.GetNumberOfPoints()-1):
				pts.GetPoint(ptId, p0)
				pts.GetPoint(ptId+1, p1)
				dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
			appendFilter.AddInputConnection(dijkstra.GetOutputPort())
		Length_geo_ctrl=dist/10
		appendFilter.Update()
		dist_geodesic_ctrl_procs.append(Length_geo_ctrl)
	
	progressBar.value = 100
	
	#if Cz is not in the middle: Error
	Delta_error=0.5 #arbitrary decision: 0.5cm
	logic = GeodesicSlicerLogic()
	if (abs(dist_geodesic_ctrl_procs[0]-dist_geodesic_ctrl_procs[1])>=Delta_error) or (abs(dist_geodesic_ctrl_procs[2]-dist_geodesic_ctrl_procs[3])>=Delta_error):
		error_text='Cz was not exactly in the middle of the sclap: Consequence, poor placement\nNot use the "Project the stimulation site" module'
		slicer.util.errorDisplay(error_text, windowTitle='Geodesic Slicer error', parent=None, standardButtons=None)
		progressBar.close()		
		return False
	
	logging.info('Processing completed')
		
	return True
	
  def setStimulationPoint(self, switch):
    self.StimulationPoint = switch
	
  def setM1Site(self, switch):
    self.M1Site = switch
	
  def split_list(self, alist, wanted_parts):
	length = len(alist)
	return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
			 for i in range(wanted_parts) ]
	
  def ProjectedPoint(self, fiducialInput, inputModel, lengthOutput2, lengthOutput3, lengthOutput4):
	"""
	Project the stimulation site, the minimum euclidian distance between the scalp and the stimulation site
	"""
	##################
	####1.dist min####
	##################
	
	logging.info('Processing started')
	
	# wait popup
	progressBar=slicer.util.createProgressDialog()
	#slicer.app.processEvents()
	
	#Access to Fiducial Properties
	fidNode = fiducialInput.currentNode()
	target_point = [0,0,0]
	fidNode.GetNthFiducialPosition(fidNode.GetNumberOfFiducials()-1,target_point) #the last fiducial

	#list of all mesh point
	all_mesh_points=[]
	for i in xrange(0, inputModel.GetPolyData().GetNumberOfPoints()):
	  p=inputModel.GetPolyData().GetPoint(i)
	  all_mesh_points.append(p)
		
	#split list top[1]/bottom[0]
	half_list=self.split_list(all_mesh_points,2) #split in two to speed up the processus
	end=half_list[1]

	#list left right:
	left, right = [], []
	#end=end[1:6]
	index = 0
	for f in end:
	  #print f,index,end[index][0]
	  if end[index][0]<0 :
		left.append(f)
	  else:
		right.append(f)
	  index += 1

	the_list=[]
	the_list=right if target_point[0] >= 0 else left
	
	#calculate the stimulation point scalp-to cortex distance
	a = numpy.array((target_point[0],target_point[1],target_point[2])) #coordinate of the target point
	dist_list=[]
	for i in the_list:
	  b = numpy.array((i[0],i[1],i[2]))
	  dista = numpy.linalg.norm(a-b) #know the distance 
	  dist_list.append(dista)

	#stim point
	pos=dist_list.index(min(dist_list))  #connaitre position de distance minimum
	stim_point=the_list[pos]

	stim_point_label=slicer.modules.markups.logic().AddFiducial(stim_point[0],stim_point[1],stim_point[2])
	fidNode.SetNthFiducialLabel(stim_point_label, "stim_point")
	
	#########################
	####nearest electrode####
	#########################

	#list electrode depuis 10-20eeg
	#numberFids = fiducialInput.currentNode().GetNumberOfFiducials()
	numberFids = fidNode.GetNumberOfFiducials()
	EEG_electrodes=[]
	EEG_electrodes_names=[]
	for i in range(numberFids):
		ras1 = [0,0,0]
		fidNode.GetNthFiducialPosition(i,ras1)
		EEG_electrodes.append(ras1)
		EEG_electrodes_names.append(fidNode.GetNthFiducialLabel(i))

	EEG_electrodes_real=EEG_electrodes[4:-3] #remove nasion,inion,tragus left & right, stim_point & T3P3
	EEG_electrodes_names_real=EEG_electrodes_names[4:-3] #remove nasion,inion,tragus left & right, stim_point & T3P3
	stim=numpy.array((stim_point[0],stim_point[1],stim_point[2])) #coordinate of the stim point
	
	dist_stim_EEG=[]
	for i in EEG_electrodes_real:
	  eeg = numpy.array((i[0],i[1],i[2]))
	  dist_eeg = numpy.linalg.norm(stim-eeg) #know the distance 
	  dist_stim_EEG.append(dist_eeg)

	dist_stim_EEG_3lowest=nsmallest(3,dist_stim_EEG) #return a list of the 3 lowest values in another list
	pos1=dist_stim_EEG.index(dist_stim_EEG_3lowest[0]) #know their position
	pos2=dist_stim_EEG.index(dist_stim_EEG_3lowest[1]) #know their position
	pos3=dist_stim_EEG.index(dist_stim_EEG_3lowest[2]) #know their position

	#print 3nearest electrode
	print EEG_electrodes_names_real[pos1],EEG_electrodes_names_real[pos2],EEG_electrodes_names_real[pos3]

	#Geodesicdistance with the 3 nearest electrodes
	#locator
	pd = inputModel.GetModelDisplayNode()
	pd1=pd.GetOutputPolyData()
	pd1.GetNumberOfPoints()
	loc = vtk.vtkPointLocator()
	loc.SetDataSet(pd1)
	loc.BuildLocator()
	vertex_stim = loc.FindClosestPoint(stim_point) 
	vertex_electrode1 = loc.FindClosestPoint(EEG_electrodes_real[pos1]) 
	vertex_electrode2 = loc.FindClosestPoint(EEG_electrodes_real[pos2]) 
	vertex_electrode3 = loc.FindClosestPoint(EEG_electrodes_real[pos3]) 

	#get the distance of the geodesic path 
	appendFilter = vtk.vtkAppendFilter()
	appendFilter.MergePointsOn()
	points = vtk.vtkPoints()

	#begin
	vIds_electrode1 = [vertex_stim,vertex_electrode1]
	vIds_electrode2 = [vertex_stim,vertex_electrode2]
	vIds_electrode3 = [vertex_stim,vertex_electrode3]
	list_vIds=[vIds_electrode1,vIds_electrode2,vIds_electrode3]

	progressBar.value = 40
	
	dist_geodesic_stim_EEG_3lowest=[]
	for nu in range(len(list_vIds)):
		vIds=list_vIds[nu]
		p0 = [0,0,0]
		p1 = [0,0,0]
		dist = 0.0
		
		m = nu +1
		#print m
		if m > 0:
			progressBar.value = 60-(60/m)
			slicer.app.processEvents()
			
		#print vIds
		for n in range(len(vIds)-1):
			v0 = vIds[n]
			v1 = vIds[n+1]
			#create geodesic path: vtkDijkstraGraphGeodesicPath
			dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
			dijkstra.SetInputConnection(pd.GetOutputPolyDataConnection())
			dijkstra.SetStartVertex(v0)
			dijkstra.SetEndVertex(v1)
			dijkstra.Update()
			pts = dijkstra.GetOutput().GetPoints()
			end = n<len(vIds)-2 and 0 or -1
			for ptId in range(pts.GetNumberOfPoints()-1, end, -1):
				pts.GetPoint(ptId, p0)
				points.InsertNextPoint(p0)		
			for ptId in range(pts.GetNumberOfPoints()-1):
				pts.GetPoint(ptId, p0)
				pts.GetPoint(ptId+1, p1)
				dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
			appendFilter.AddInputConnection(dijkstra.GetOutputPort())
		Length_geo=dist/10
		appendFilter.Update()
		dist_geodesic_stim_EEG_3lowest.append(Length_geo)

	# nearest electrode
	lengthOutput2.text = str(EEG_electrodes_names_real[pos1])+' at '+str(round(dist_geodesic_stim_EEG_3lowest[0],3))+' cm'
	lengthOutput3.text = str(EEG_electrodes_names_real[pos2])+' at '+str(round(dist_geodesic_stim_EEG_3lowest[1],3))+' cm'
	lengthOutput4.text = str(EEG_electrodes_names_real[pos3])+' at '+str(round(dist_geodesic_stim_EEG_3lowest[2],3))+' cm'
	
	progressBar.value = 100
	logging.info('Processing completed')
		
	return True

  def setMTunadjusted(self, newValue):
	#print "MT unadjusted2:", newValue
	self.MTunadjusted = newValue
	return self.MTunadjusted
	
  def CorrectedPoint(self, fiducialInput, inputModel, setMTunadjusted, MTOutput, MTOutput2):
	"""
	Project the stimulation site, the minimum euclidian distance between the scalp and the stimulation site
	"""
	##################
	####1.dist min####
	##################
	
	logging.info('Processing started')
	
	# wait popup
	progressBar=slicer.util.createProgressDialog()
	#slicer.app.processEvents()
	
	#Access to Fiducial Properties
	fidNode = fiducialInput.currentNode()
	target_point = [0,0,0]
	fidNode.GetNthFiducialPosition(fidNode.GetNumberOfFiducials()-3,target_point) #the stimulation point depth

	#list of all mesh point
	all_mesh_points=[]
	for i in xrange(0, inputModel.GetPolyData().GetNumberOfPoints()):
	  p=inputModel.GetPolyData().GetPoint(i)
	  all_mesh_points.append(p)
		
	#split list top[1]/bottom[0]
	half_list=self.split_list(all_mesh_points,2) #split in two to speed up the processus
	end=half_list[1]

	#list left right:
	left, right = [], []
	#end=end[1:6]
	index = 0
	for f in end:
	  #print f,index,end[index][0]
	  if end[index][0]<0 :
		left.append(f)
	  else:
		right.append(f)
	  index += 1

	the_list=[]
	the_list=right if target_point[0] >= 0 else left
	
	#calculate the stimulation point scalp-to cortex distance
	a = numpy.array((target_point[0],target_point[1],target_point[2])) #coordinate of the target point
	dist_list=[]
	for i in the_list:
	  b = numpy.array((i[0],i[1],i[2]))
	  dista = numpy.linalg.norm(a-b) #know the distance 
	  dist_list.append(dista)
	
	########################################################
	####rTMS resting motor threshold-- Correction factor####
	########################################################

	#place the M1 Point The primary motor cortex (Brodmann area 4)
	M1_point = [0,0,0]
	fidNode.GetNthFiducialPosition(fidNode.GetNumberOfFiducials()-1,M1_point) #the last fiducial
	 
	#calculate the M1 scalp-to-cortex distance
	a = numpy.array((M1_point[0],M1_point[1],M1_point[2])) #coordinate of the target point
	dist_list_M1=[]
	for i in the_list:
	  b = numpy.array((i[0],i[1],i[2]))
	  distance_M1 = numpy.linalg.norm(a-b) #know the distance 
	  dist_list_M1.append(distance_M1)
	  
	DM1=min(dist_list_M1)

	#MT (made directly in geodesicSlicer)
	MTInput=setMTunadjusted
	
	#According to Stokes et al. Clin  Neurophysiol 2007
	# AdjMT% = MT + 2.8*(DsiteX -DM1)
	DsiteX = min(dist_list)
	#print DsiteX,DM1,MTInput
	
	AdjMT = MTInput + 2.8*(DsiteX - DM1)
	
	#According to Hoffman et al. Biol Psychiatry 2013
	#[Adjusted Stimulation Strength = 0.90*rMT*e0.036*(SCDt-SCDm)]
	SCDt= DsiteX
	SCDm= DM1
	rMT= MTInput
	print ('SCDx= ',SCDt,'SCDm= ',SCDm,'rMT= ',rMT)
	ASS= 0.90*rMT*math.exp(0.036*(SCDt-SCDm))
	
	# Correction factor
	MTOutput.text = str(AdjMT)
	MTOutput2.text = str(ASS)
		
	progressBar.value = 100
	logging.info('Processing completed')
	
	return True
