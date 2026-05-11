#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  vismol_object.py
#  
#  Copyright 2022 Carlos Eduardo Sequeiros Borja <casebor@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import time
import numpy as np
from logging import getLogger
from vismol.model.atom import Atom
from vismol.model.bond import Bond
from vismol.model.chain import Chain
from vismol.model.residue import Residue
from vismol.model.molecule import Molecule
#from vismol.model.molecular_properties import COLOR_PALETTE
from vismol.libgl.vismol_font import VismolFont
from vismol.libgl.representations import DotsRepresentation
from vismol.libgl.representations import LinesRepresentation
from vismol.libgl.representations import NonBondedRepresentation
from vismol.libgl.representations import PickingDotsRepresentation
from vismol.libgl.representations import ImpostorRepresentation
from vismol.libgl.representations import SticksRepresentation
from vismol.libgl.representations import SpheresRepresentation
from vismol.libgl.representations import DashedLinesRepresentation
from vismol.libgl.representations import LabelRepresentation
from vismol.libgl.representations import SurfaceRepresentation
from vismol.libgl.representations import CartoonRepresentation
# from vismol.libgl.representations import WiresRepresentation
# from vismol.libgl.representations import RibbonsRepresentation
import vismol.utils.c_distances as cdist

logger = getLogger(__name__)


class VismolObject:
    """ Visual Object contains the information necessary for openGL to draw 
        a model on the screen. Everything that is represented in the graphical 
        form is stored in the form of a VismolObject.
        
        
        Arguments:
        - vismol_session: Vismol Session - Necessary to build the "atomtree_structure".
                          The vismol_session contains the atom_id_counter (self.vm_session.atom_id_counter).
        - index: Unique index for the VismolObject to find it in self.vm_session.vismol_objects_dic.
        - name: Label that describes the object (default: "UNK").
        - active: Boolean flag to enable/disable the object (default: False).
        - trajectory: A list of coordinates representing the trajectory frames.
                      Each frame is represented as a list of [x, y, z] coordinates (optional).
        - color_palette: Integer number to access the color pick for carbon atoms (0 = green, 1 = purple, ...) (optional).
        - bonds_pair_of_indexes: Pair of atoms used to define bonds, like [1, 3, 1, 17, 3, 4, 4, 20] (optional).
   
        
        Arguments
        
        name       = string  - Label that describes the object  
        atoms      = list of atoms  - [index, at_name, cov_rad,  at_pos, at_res_i, at_res_n, at_ch]
        vismol_session  = Vismol Session - Necessary to build the "atomtree_structure"
                     vismol_session contains the atom_id_counter (self.vm_session.atom_id_counter)
        trajectory = A list of coordinates - eg [ [x1,y1,z1, x2,y2,z2...], [x1,y1,z1, x2,y2,z2...]...]
                     One frame is is required at last.
    """
    def __init__(self, vismol_session, index, name="UNK", active=False, trajectory=None,
                 color_palette=None, bonds_pair_of_indexes=None):
        """ Class initialiser """
        # References to vismol_session and its configuration
        self.vm_session = vismol_session
        self.vm_config = vismol_session.vm_config
                                   
                                   # Unique index for the VismolObject to find it in self.vm_session.vismol_objects_dic
        self.index = index         # import to find vboject in self.vm_session.vismol_objects_dic
        
        self.name = name
        self.active = active       # Boolean flag to enable/disable the object (show and hide) # for "show and hide"   enable/disable
        
        self.frames = trajectory  # A list of coordinates representing the trajectory frames
                                  # Each frame is represented as a list of [x, y, z] coordinates
                                  # (set as None if not provided)
        
        if color_palette is None:
            self.color_palette = self.vm_session.periodic_table.get_color_palette() #None #COLOR_PALETTE[0] # Default color palette for carbon atoms (0 = green)
        else:
            self.color_palette = color_palette #this is an integer num to access the color pick for carbon atoms (0 = green, 1 = purple, ...) 
                                               # Integer number to access the specified color palette
        
                                   
        self.editing = False       # for translate and rotate  xyz coords
                                   # Boolean flag for translate and rotate XYZ coordinates 
        
        self.mass_center = np.zeros(3, dtype=np.float32)
        self.vm_font = VismolFont() # Font object used for visualization
        self.coords = None          # Set as None for now (not defined in the provided code)
        
        
        # Dictionaries to store atoms, residues, chains, and molecules with their respective indexes as keys
        self.atoms = {}
        #self.residues = {}
        self.chains = {}
        self.molecules = {}

        self.atom_unique_id_dic = {}   # Dictionary to store atom unique identifiers (not yet defined in the code)
        self.selected_atom_ids = set() # Set to store the IDs of selected atoms (not yet defined in the code)
        
        
        self.bonds = None       # Bond objects representing connections between atoms
        self.index_bonds = None # Pair of atoms, something like: [1, 3, 1, 17, 3, 4, 4, 20]
                                # Pair of atoms used to define bonds (set as None if not provided)
        
        self.bond_order_list = None  # A bond order list [1,1,1,2,1,1, and so on...]
        
        self.non_bonded_atoms = None # Array of indexes
                                     # Array of indexes for non-bonded atoms (not yet defined in the code)
                                     
        self.cov_radii_array = None  # a list of covalent radius values for all  --> will be used to calculate de bonds
                                     # List of covalent radius values for all atoms (not yet defined in the code)
        
        self.electronegativity_array = None # a list Electronegativity from UFF force field
                                            # see ../utils/elements.py , colunm 8 - En_UFF
          
        
        self.topology = {} # {92: [93, 99], 93: [92, 94, 96, 100], 99: [92],...}
                           # important to define molecules
        
        self.rings = [] # [[1,2,3,4], [5,5,6,7], ...] each element is list of indexes and a ring
                           
        self.representations = {}# Dictionary to store different visualization representations for the object
        # (initialized with None values for each representation type)
        
        for rep_type in self.vm_config.representations_available:
            self.representations[rep_type] = None
        self.representations['ribbon_sphere'] = None
        self.representations['stick_spheres'] = None
        
        # Additional transformation matrices for the object's visual representation
        self.model_mat = np.identity(4, dtype=np.float32)
        self.trans_mat = np.identity(4, dtype=np.float32)
        
        self.core_representations = {"picking_dots":None, "picking_text":None} # Core representation objects
        self.selection_dots_vao = None
        self.selection_dot_buffers = None
        self.picking_dots_vao = None
        self.picking_dot_buffers = None
        
        self.dynamic_bonds  = [] # Pair of atoms, something like: [[0,1,1,2,3,4] , [0,1,1,2], ...]
                                # Like self.index_bonds but for each frame
        
        self.c_alpha_bonds = [] # List of pair of atoms defining dynamic bonds for each frame
        self.c_alpha_atoms = [] # List of pair of atoms defining C-alpha bonds
    
        ''' Cell and Symmetry'''
        # Cell and Symmetry attributes (not yet defined in the code)
        self.cell_parameters  = None
        self.cell_coordinates = None
        self.cell_indexes     = None
        self.cell_colors      = None
        self.cell_bonds       = None
        self.representations['labels'] = None
    
    
    def build_core_representations(self):
        """
        Function doc: Builds core representations for the VismolObject.

        Note: The function creates core representation objects for the VismolObject and
        updates the 'core_representations' dictionary with the created representations.
        """
        
        # Create a PickingDotsRepresentation object and add it to the 'core_representations' dictionary
        self.core_representations["picking_dots"] = PickingDotsRepresentation(self,
                                                    self.vm_session.vm_glcore, active=True,
                                                    indexes=list(self.atoms.keys()))
        
        # Create a DashedLinesRepresentation object and add it to the 'core_representations' dictionary
        self.core_representations["dash"] = DashedLinesRepresentation(self, self.vm_session.vm_glcore,
                                                active=True, indexes=self.index_bonds)
    
    def create_representation(self, rep_type="lines", indexes=None):
        """   
        Function doc: Creates a visualization representation for the VismolObject.

        Parameters:
        - rep_type: The type of representation to create.
        - indexes: List of atom indexes or bond indexes based on the representation type (optional).
    
        Note: The function creates a new visualization representation object for the VismolObject
        and updates its internal 'representations' dictionary. It initializes the representation
        according to the provided 'rep_type' and other relevant parameters.
        """
        # Create a DotsRepresentation object and add it to the 'representations' dictionary
        if rep_type == "dots":
            self.representations["dots"] = DotsRepresentation(self, self.vm_session.vm_glcore,
                                                active=True, indexes=list(self.atoms.keys()))
        elif rep_type == "lines":
            # Create a LinesRepresentation object and add it to the 'representations' dictionary
            self.representations["lines"] = LinesRepresentation(self, self.vm_session.vm_glcore,
                                                active=True, indexes=self.index_bonds)
        elif rep_type == "nonbonded":
            # Create a NonBondedRepresentation object and add it to the 'representations' dictionary
            self.representations["nonbonded"] = NonBondedRepresentation(self, self.vm_session.vm_glcore,
                                                    active=True, indexes=self.non_bonded_atoms)
        elif rep_type == "impostor":
            # Create an ImpostorRepresentation object and add it to the 'representations' dictionary
            self.representations["impostor"] = ImpostorRepresentation(self, self.vm_session.vm_glcore,
                                                active=True, indexes=list(self.atoms.keys()))
        elif rep_type == "sticks":
            # Create a SticksRepresentation object and add it to the 'representations' dictionary
            self.representations["sticks"] = SticksRepresentation(self, self.vm_session.vm_glcore,
                                                                  active=True, indexes=self.index_bonds)
        elif rep_type == 'stick_spheres':
            # Create a SpheresRepresentation object and add it to the 'representations' dictionary
            self.representations['stick_spheres'] = SpheresRepresentation(self, self.vm_session.vm_glcore,
                                                    active=True, indexes=list(self.atoms.keys()), mode = 3)
                           
                                                                  
        elif rep_type == "spheres":
            # Create a SpheresRepresentation object and add it to the 'representations' dictionary
            self.representations["spheres"] = SpheresRepresentation(self, self.vm_session.vm_glcore,
                                                    active=True, indexes=list(self.atoms.keys()) )
        
        elif rep_type == "picking_spheres":
            # Create a SpheresRepresentation object with picking mode and add it to the 'representations' dictionary
            self.representations["picking_spheres"] = SpheresRepresentation(self, self.vm_session.vm_glcore,
                                                    active=True, indexes=list(self.atoms.keys()), mode =1 )
        
        elif rep_type == "vdw_spheres":
            # Create a SpheresRepresentation object with Van der Waals radius and add it to the 'representations' dictionary
            self.representations["vdw_spheres"] = SpheresRepresentation(self, self.vm_session.vm_glcore,
                                                    active=True, indexes=list(self.atoms.keys()), vdw =True)

        elif rep_type == "dash":
            # Create a DashedLinesRepresentation object and add it to the 'representations' dictionary
            self.representations["dash"] = DashedLinesRepresentation(self, self.vm_session.vm_glcore,
                                                active=True, indexes=self.index_bonds)
        elif rep_type == "ribbons":
            # Create a SticksRepresentation object with 'ribbons' name and add it to the 'representations' dictionary
            self.representations["ribbons"] = SticksRepresentation(self, self.vm_session.vm_glcore,
                                                                   active=True, indexes=self.index_bonds, name  = 'ribbons')
        elif rep_type == 'ribbon_sphere':
            # Create a SpheresRepresentation object and add it to the 'representations' dictionary
            self.representations['ribbon_sphere'] = SpheresRepresentation(self, self.vm_session.vm_glcore,
                                                    active=True, indexes=list(self.atoms.keys()), mode = 2  )
                                                                    
        elif rep_type == "dynamic":
            # Create a SticksRepresentation object with 'dynamic' flag and add it to the 'representations' dictionary
            #print(self.dynamic_bonds)
            self.representations["dynamic"] = SticksRepresentation(self, self.vm_session.vm_glcore,
                                                                  active=True, indexes=self.index_bonds, is_dynamic = True)
        
        elif rep_type == "labels":
            # Create a LabelRepresentation object and add it to the 'representations' dictionary
            self.representations["labels"] = LabelRepresentation(vismol_object  = self  ,  
                                                                  vismol_glcore = self.vm_session.vm_glcore , 
                                                                  indexes       = [0,1,2] , 
                                                                  labels        = None     , 
                                                                  color         = [1, 1, 0, 1])
        elif rep_type == "surface":
            # Create a LabelRepresentation object and add it to the 'representations' dictionary
            self.representations["surface"] = SurfaceRepresentation(vismol_object = self                      ,
                                                                    vismol_glcore = self.vm_session.vm_glcore ,  
                                                                    name          = 'surface'                 ,
                                                                    active        = True                      ,
                                                                    indexes       = []                        ,
                                                                    is_dynamic    = False)  
            
            
            #self.representations["surface"] = SurfaceRepresentation(vismol_object = self  ,
            #                                                        vismol_glcore = self.vm_session.vm_glcore)
                                                                    #rep_type      = "mol", 
                                                                    #vismol_object = self, 
                                                                    #vm_glcore     = self.vm_session.vm_glcore, 
                                                                    #indexes       = [])
            
            #self.representations["surface"] = SurfaceRepresentation(vismol_object  = self  ,  
            #                                                      vismol_glcore = self.vm_session.vm_glcore , 
            #                                                      indexes       = [0,1,2] , 
            #                                                      labels        = None     , 
            #                                                      color         = [1, 1, 0, 1])
        
        
        elif rep_type == 'cartoon':
            self.representations["cartoon"] =CartoonRepresentation(name = 'cartoon', active = True, 
                                                                   rep_type = 'mol', vismol_object = self, 
                                                                   vismol_glcore = self.vm_session.vm_glcore, 
                                                                   indexes = [])
        
        # elif rep_type == "dotted_lines":
        #     self.representations["dotted_lines"] = LinesRepresentation(self, self.vm_session.vm_glcore,
        #                                                                active=True, indexes=indexes)
        else:
            # Handle error when an unsupported representation type is provided
            logger.error("Representation {} not implemented".format(rep_type))
            raise NotImplementedError("Representation {} not implemented".format(rep_type))
    
    def _generate_color_vectors(self, colors_id_start, do_colors_raindow=True,
                                do_vdw_dot_sizes=True, do_cov_dot_sizes=True):
        """ (1) This method assigns to each atom of the system a 
            unique identifier based on the RGB color standard. 
            This identifier will be used in the selection function. 
            There are no two atoms with the same color ID in  
            
            (2) This method builds the "colors" np array that will 
            be sent to the GPU and which contains the RGB values 
            for each atom of the system.


            Parameters:
            - colors_id_start: The starting value for color IDs, which determines the color scheme for atoms.
            
            - do_colors_rainbow: If True, generates a rainbow color scheme based on the atom's position in the atom list.
                                 If False, the atom colors will be based on the atom's individual color attribute.
            
            - do_vdw_dot_sizes: If True, generates Van der Waals dot sizes based on the atom's Van der Waals radius.
            
            - do_cov_dot_sizes: If True, generates covalent dot sizes based on the atom's covalent radius.

            Note: The function updates several attributes of the VismolObject, including 'colors', 'color_indexes',
                  'color_rainbow', 'vdw_dot_sizes', and 'cov_dot_sizes', based on the provided parameters.
        """
        atom_qtty = len(self.atoms)
        half = int(atom_qtty/2)
        quarter = int(atom_qtty/4)
        color_step = 1.0/(atom_qtty/4.0)
        red = 0.0
        green = 0.0
        blue = 1.0
        
        self.colors = np.empty([len(self.atoms), 3], dtype=np.float32)
        self.color_indexes = np.empty([len(self.atoms), 3], dtype=np.float32)
        if do_colors_raindow:
            self.color_rainbow = np.empty([len(self.atoms), 3], dtype=np.float32)
        if do_vdw_dot_sizes:
            self.vdw_dot_sizes = np.empty(len(self.atoms), dtype=np.float32)
        if do_cov_dot_sizes:
            self.cov_dot_sizes = np.empty(len(self.atoms), dtype=np.float32)
        
        for i, atom in self.atoms.items():
            self.colors[i] = atom.color
            self.color_indexes[i] = atom.color_id
            if do_vdw_dot_sizes: 
                self.vdw_dot_sizes[i] = atom.vdw_rad * 3
            if do_cov_dot_sizes: 
                self.cov_dot_sizes[i] = atom.cov_rad
            if do_colors_raindow:
                if i <= 1*quarter:
                    self.color_rainbow[i,:] = red, green, blue
                    green += color_step
                
                if (i >= 1*quarter) and (i <= 2*quarter):
                    self.color_rainbow[i,:] = red, green, blue
                    blue -= color_step
                
                if (i >= 2*quarter) and (i <= 3*quarter):
                    self.color_rainbow[i,:] = red, green, blue
                    red += color_step
                
                if (i >= 3*quarter) and (i <= 4*quarter):
                    self.color_rainbow[i,:] = red, green, blue
                    green -= color_step
    
    def define_bonds_from_external (self, index_bonds = [], internal = True):
        """ Function doc """
        if internal is True:
            self.index_bonds = index_bonds

            self._bonds_from_pair_of_indexes_list()
            self._get_non_bonded_from_bonded_list()
            
            self._generate_topology_from_index_bonds()
            #self.find_rings(self.topology)
            self.define_molecules()
            self.define_Calpha_backbone()
        else:
            return index_bonds
    
    def find_bonded_and_nonbonded_atoms(self, selection=None, frame=0, gridsize=1.2,
                                         maxbond=2.4, tolerance=1.2, internal = True, debug = False):
        """
        Function doc: Determines bonded and nonbonded atoms based on selection in a VismolObject.
        
        Parameters:
        - selection: A dictionary containing the selected atoms (optional).
        - frame: Frame number for the atom coordinates (default is 0).
        - gridsize: Grid size for bond calculation (usually should not be changed, default is 0.8).
        - maxbond: Size of the maximum bond to be monitored (bonds greater than maxbond may be disregarded, default is 2.4 Å).
        - tolerance: Safety factor that multiplies the term (ra_cov + rb_cov)**2 (default is 1.4).
        - internal: If True, calculates the bindings for the object itself (used in the object's genesis).
                    If False, returns a list of atom_ids representing the bonds between atoms.

        Returns:
        If internal is True, the function updates the VismolObject's internal attributes:
        - index_bonds: List of pairs of atom indexes representing the bonded atoms.
        - bonds: List of Bond objects representing the bonds between atoms.
        - topology: Dictionary representing the atom topology and connectivity.
        - Non-bonded interactions and other attributes are also updated.

        If internal is False, the function returns a list of atom_ids representing the bonds between atoms.

        Note: This function calculates bonds between atoms based on their positions and covalent radii.
        
            Receives a dictionary as a selection:

            selection = {'atom_id': atom_object, ...} ()
            
            frame - (frame number)
            
            grid_size - (usually should not be changed. Default is: 0.8)
            
            maxbond - (Size of the maximum bond to be monitored. Bonds greater than "maxbond" 
            may be disregarded. Default is: 2.4 A)

            tolerance (Safety factor that multiplies the term (ra_cov + rb_cov)**2. Default is: 1.4)

            internal (If True, calculates the bindings for the object itself. Used in the object's 
            
            genesis. If False, returns a list of atom_ids like  [0,1, # bond  between 0 and 1 
                                                                 1,2, # bond  between 1 and 2
                                                                 0,3] # bond  between 0 and 3 )
        

        """
        # Check if internal is True and there is already information about contacts.
        if internal:
            if self.index_bonds is not None:
                logger.critical("It seems that there is already information about "\
                    "the contacts in this VismolObject, trying to override the data "\
                    "can produce serious problems :(")
        
        
        # Initialize variables and data structures
        initial = time.time()
        atoms_frame_mask = np.zeros(len(self.atoms), bool)
        if self.cov_radii_array is None:
            
            self.cov_radii_array = np.empty(len(self.atoms), dtype=np.float32)
            self.electronegativity_array = np.empty(len(self.atoms), dtype=np.float32)
            
            for i, atom in self.atoms.items():
                self.cov_radii_array[i] = atom.cov_rad
                self.electronegativity_array[i] = atom.electronegativity
                
        # Create a mask to identify atoms in the frame (all atoms if selection is None)
        if selection is None:
            selection = self.atoms
            atoms_frame_mask[:] = True
        else:
            atoms_frame_mask[:] = False
            for atom in selection.values():
                atoms_frame_mask[atom.atom_id] = True
        
        
        # Extract relevant data for bond calculation
        #cov_rads = self.cov_radii_array[atoms_frame_mask]
        #coords = self.frames[frame][atoms_frame_mask]
        cov_rads = self.cov_radii_array
        coords = self.frames[frame]#[atoms_frame_mask]
        
        
        # Generate indexes and grid positions for each atom in the selection
        indexes = []
        gridpos_list = []
        for atom in selection.values():
            indexes.append(atom.atom_id)
            gridpos_list.append(atom.get_grid_position(gridsize=gridsize, frame=frame))
        #print(  'grid elementes:',gridpos_list)
        if debug:
            logger.debug("Time used for preparing the atom mask, covalent radii list "\
                         "and grid positions: {}".format(time.time() - initial))
        
        
        # Calculate bonds based on grid positions and covalent radii
        if internal is True:
            initial = time.time()
            self.index_bonds = cdist.get_atomic_bonds_from_grid(indexes, coords,
                                            cov_rads, gridpos_list, gridsize, maxbond, tolerance)
            msg = """Building grid elements  :
        Total number of Atoms   : {}
        Gridsize                : {}
        Bonds                   : {}
        Bonds calcultation time : {} seconds""".format(len(selection), gridsize,
                                    len(self.index_bonds), time.time() - initial)
            logger.info(msg)
            
            # Create Bond objects and update atom bond lists
            self._bonds_from_pair_of_indexes_list()
            
            # Generate non-bonded interactions from bonded atoms
            self._get_non_bonded_from_bonded_list()
            
            # Generate atom topology from index_bonds
            #bachega's code
            initial = time.time()
            self._generate_topology_from_index_bonds()
            
            # Define molecules based on the atom topology
            self.define_molecules()
            
            
            final = time.time()
            print('        Defining molecule indexes: ', final - initial)
            
            # Define Calpha backbone atoms
            self.define_Calpha_backbone()

        else:            
            # Calculate bonds based on grid positions and covalent radii and return the results
            index_bonds = cdist.get_atomic_bonds_from_grid(indexes, coords,
                                            cov_rads, gridpos_list, gridsize, maxbond, tolerance)
            
#            msg = """Building grid elements  :
#        Total number of Atoms   : {}
#        Gridsize                : {}
#        Bonds                   : {}
#        Bonds calcultation time : {} seconds""".format(len(selection), gridsize,
#                                    len(index_bonds), time.time() - initial)
#            logger.info(msg)
            
                       
            # Dynamic bonds is not working properly for pure QC system
            #'''
            bonds = index_bonds
            new_bonds = []
            for index in range (0,len(index_bonds), 2):
                if [bonds[index], bonds[index+1]] in new_bonds or [bonds[index+1], bonds[index]] in new_bonds:
                    pass
                else:
                    new_bonds.append([bonds[index], bonds[index+1]])
            #print(new_bonds)
            index_bonds = [item for sublista in new_bonds for item in sublista]
            #'''
            return index_bonds
    
    def _get_covalent_radii (self):
        """ Function doc """
        if self.cov_radii_array is None:
            self.cov_radii_array = np.empty(len(self.atoms), dtype=np.float32)
            for i, atom in self.atoms.items():
                self.cov_radii_array[i] = atom.cov_rad

    def _bonds_from_pair_of_indexes_list(self, exclude_list = [['H','H']]):
        """ 
        Creates Bond objects based on pairs of indexes in self.index_bonds list.
        The bonds list is populated with the created Bond objects, and each
        atom involved in a bond is updated with the respective Bond object.
        
        self.index_bonds = [0,1  , 0,4  ,  1,3  , ...]
        self.bonds = [bond1(obj), bond2(obj), bond3(obj), ...] 
        """
        assert self.bonds is None # Ensure the bonds list is not already initialized
        self.bonds = [] # Initialize an empty list to store the Bond objects
        self.bond_order_list =[] # Initialize an empty list to store the Bond orders
        new_index_bonds = []
        # Loop through the self.index_bonds list in pairs
        for i in range(0, len(self.index_bonds)-1, 2):
            
            index_i = self.index_bonds[i]    # Get the first atom's index of the bond
            index_j = self.index_bonds[i+1]  # Get the second atom's index of the bond
            
            is_excluded = False
            
            for excluded_bond in exclude_list: 
                if self.atoms[index_i].symbol in excluded_bond and self.atoms[index_j].symbol in excluded_bond:
                    is_excluded = True
            
            
            if is_excluded:
                pass
            else:
                new_index_bonds.append(index_i)
                new_index_bonds.append(index_j)
            
                #index_i = self.index_bonds[i]    # Get the first atom's index of the bond
                #index_j = self.index_bonds[i+1]  # Get the second atom's index of the bond
                
                
                
                # Create a Bond object with the atoms and their indexes
                bond = Bond(atom_i=self.atoms[index_i], atom_index_i=index_i,
                            atom_j=self.atoms[index_j], atom_index_j=index_j)
                
                
                bond.get_bond_order()
                self.bond_order_list.append(bond.bond_order)
                
                
                # Add the created Bond object to the bonds list
                self.bonds.append(bond)
                
                # Update the atoms with the created Bond object, indicating their bond connections
                self.atoms[index_i].bonds.append(bond)
                self.atoms[index_j].bonds.append(bond)
        
        # Convert the index_bonds list to a numpy array of unsigned 32-bit integers
        self.index_bonds = new_index_bonds
        #print(self.index_bonds)
        self.index_bonds     = np.array(self.index_bonds, dtype=np.uint32)
        self.bond_order_list = np.array(self.bond_order_list, dtype=np.uint32)
        #self.bond_order_list = np.repeat(self.bond_order_list, 2)
    def _get_non_bonded_from_bonded_list(self):
        """ Function doc """
        assert self.non_bonded_atoms is None
        bonded_set = set(self.index_bonds)
        self.non_bonded_atoms = []
        for i, atom in self.atoms.items():
            if i in bonded_set:
                atom.nonbonded = False
            else:
                atom.nonbonded = True
                self.non_bonded_atoms.append(i)
        self.non_bonded_atoms.sort()
        self.non_bonded_atoms = np.array(self.non_bonded_atoms, dtype=np.int32)
    
    def _get_center_of_mass(self, frame=0):
        """ Function doc """
        if frame >= self.frames.shape[0]:
            logger.info("Frame {} is out of range for trajectory of size {}. \
                Using the last frame.".format(frame, self.frames.shape[0] - 1))
            frame = self.frames.shape[0] - 1
        return np.mean(self.frames[frame], axis=0)
    
    def set_model_matrix(self, mat):
        """ Function doc """
        self.model_mat = np.copy(mat)
    
    def _generate_topology_from_index_bonds(self, bonds = None):
        """ 
        bonds = [92,93  ,  92,99  ,  ...]
        
        Returns a graph in dictionary form (this may in turn be 
        needed to determine which objects are molecules).
        
        defines: self.topology = {92: [93, 99], 93: [92, 94, 96, 100], 99: [92],...}
        
        """
        if bonds is None:
            bonds =  self.index_bonds
        else:
            pass

        
        bonds_pairs = []
        topology    = {}
        
        for i  in range(0, len(bonds),2):
            bonds_pairs.append([bonds[i], bonds[i+1]])
        
        
        for bond in bonds_pairs:
            if bond[0] in topology.keys():
                topology[bond[0]].append(bond[1])
            else:
                topology[bond[0]] = []
                topology[bond[0]].append(bond[1])
            
            
            if bond[1] in topology.keys():
                topology[bond[1]].append(bond[0])
            else:
                topology[bond[1]] = []
                topology[bond[1]].append(bond[0])
        self.topology = topology
        #self.find_rings(topology)
        
        
    def find_rings(self, graph):
        '''
        The algorithm traverses all paths using DFS (depth-first search).
        A ring is identified when we return to the starting node after passing through at least two other nodes (len(path) > 2).
        After the search, it removes duplicates based on the set of nodes in each ring.
        Expected output:

        csharp
        Copy
        Edit
        Cycles found:
        [92, 93, 99]
        [93, 96, 100]
        '''
        rings = []

        def dfs(current, start, path, visited):
            path.append(current)
            visited.add(current)

            for neighbor in graph[current]:
                if neighbor == start and len(path) > 2:
                    ring = sorted(path)
                    if ring not in rings:
                        rings.append(ring[:])
                elif neighbor not in visited:
                    dfs(neighbor, start, path, visited)

            path.pop()
            visited.remove(current)

        for node in graph:
            dfs(node, node, [], set())

        # Eliminate duplicates (same cycle in different orders)
        unique_rings = []
        for ring in rings:
            if not any(set(ring) == set(c) for c in unique_rings):
                unique_rings.append(ring)
        self.rings = unique_rings
        return unique_rings

    def define_molecules (self):
        """ Function doc 
        self.topology  = It is a graph written in the form of a dictionary:
                        {
                         index1 : [index2 , index3, ...], 
                         index2 : [index1 , index4, ...]
                         ...}
        
        groups = [{0, 1, 2}, {3, 4, 5, 6}, {8, 9, 7}]
        
        self.atoms =  {
                       0: <pdynamo.pDynamo2EasyHybrid.Atom object at 0x7f3323cbcdf0>, 
                       1: <pdynamo.pDynamo2EasyHybrid.Atom object at 0x7f3323cbcfd0>, 
                       2: <pdynamo.pDynamo2EasyHybrid.Atom object at 0x7f3323cbcf40>, 
                       3: <pdynamo.pDynamo2EasyHybrid.Atom object at 0x7f3323e38250>, 
                       4: <pdynamo.pDynamo2EasyHybrid.Atom object at 0x7f3323e38040>, 
                       5: ...
                       }
        
        This function populates the molecule dictionary (self.molecules), each 
        molecule object is created very similarly to the residue object
        
        """
        #try:
        #groups = find_groups(self.topology)
        groups = find_connected_components(self.topology)
        mol_index = 0
        for mol_index, group in enumerate( groups ):
            atoms = {}
            molecule = Molecule(self, name="UNK", index = mol_index)
            
            for atom_index in list(group):
                molecule.atoms[atom_index] = self.atoms[atom_index]
                self.atoms[atom_index].molecule = molecule
                
            self.molecules[mol_index] = molecule
        
        #-------------------------------------------------------------
        # non_bonded_atoms
        # Should be here, ohterwise we will have selection problems
        #-------------------------------------------------------------

        for atom_index in self.non_bonded_atoms:
            #try:
            mol_index += 1
            molecule = Molecule(self, name="UNK", index = mol_index)
            molecule.atoms[atom_index] = self.atoms[atom_index]
            self.atoms[atom_index].molecule = molecule
            self.molecules[mol_index] = molecule

            #print(self.molecules)
        #except:
        #    print('Failure to determine the list of molecules!')
            
    def define_Calpha_backbone (self):
        """ Function doc 
        Verifica quais conexões entre c_alphas são válidas.
        """
        
        
        self.c_alpha_bonds = []
        self.c_alpha_atoms = []
        
        #
        # Building the self.c_alpha_atoms dict
        # {atom_id : atom_object, ...}
        #
        for c_index, chain in self.chains.items():
            for r_index, residue in chain.residues.items():
                residue._is_protein()
                if residue.is_protein:
                    for a_index, atom in residue.atoms.items():
                        if atom.name == "CA":
                            self.c_alpha_atoms.append(atom)

        
        for i in range(1, len(self.c_alpha_atoms)):

            atom_before  = self.c_alpha_atoms[i-1]
            resi_before  = atom_before.residue.index
            index_before = atom_before.atom_id
            

            atom  = self.c_alpha_atoms[i]
            resi  = atom.residue.index
            index = atom.atom_id
            
            '''Checks whether the two residues are in sequence 
            (otherwise there is a break in the backbone structure)'''
            if resi == resi_before + 1:
                
                bond = Bond(atom_i=atom_before, atom_index_i=index_before,
                            atom_j=atom, atom_index_j=index)
                
                distance = bond.distance()
                if distance < 4.0:
                    self.c_alpha_bonds.append(bond)
        
        #print(self.c_alpha_bonds)
        #print(self.c_alpha_atoms)
        

    def _calculate_unit_cell_vertices(self, a, b, c, alpha, beta, gamma):
        '''
        Returns the list of vertices positions of 
        a box with parameters a, b, c, alpha, beta, gamma.
        '''
        
        # Convert angles to radians
        alpha_rad = np.radians(alpha)
        beta_rad  = np.radians(beta)
        gamma_rad = np.radians(gamma)

        # Calculate unit cell vectors
        v1 = np.array([a, 0, 0])
        v2 = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])
        v3_x = c * np.cos(beta_rad)
        v3_y = c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
        v3_z = np.sqrt(c**2 - v3_x**2 - v3_y**2)
        v3 = np.array([v3_x, v3_y, v3_z])

        # Define the eight vertices of the unit cell
        vertices = [
            np.array([0, 0, 0]),
            v1,
            v2,
            v2 + v1,
            v3,
            v3 + v1,
            v3 + v2,
            v3 + v2 + v1
        ]

        return vertices
    
    def set_cell (self, a, b, c, alpha, beta, gamma, color = [0.5, 0.5, 0.5]):
        """
        Assign the cell parameters to the 
        "vismol_object" object.
        
        Arguments a, b, c, alpha, beta, gamma 
        must be obtained externally.
        
        """
        
        self.cell_parameters = {'a'     : a, 
                                'b'     : b, 
                                'c'     : c, 
                                'alpha' : alpha, 
                                'beta'  : beta, 
                                'gamma' : gamma
                               }
         
        vertices = self._calculate_unit_cell_vertices(a, b, c, alpha, beta, gamma)


        '''
        The coordinates follow the same structure as the atoms, 
        they are organized in a trajectory structure with only one 
        frame.
        '''
        self.cell_coordinates = np.empty([1, 8, 3], dtype=np.float32)
        self.cell_indexes     = []
       
        
        '''
        The connections between the vertices of the boxes 
        are pre-established.
        '''
        self.cell_bonds       = [0,1, 0,2, 2,3, 0,4 , 1,3 , 1,5 , 2,3 , 2,6 , 3,7 , 4,6 , 4,5, 5,7, 6,7]       
        self.cell_bonds       = np.array(self.cell_bonds, dtype=np.uint32)
        
        
        '''
        The colors of the vertices follow the same 
        structure used in atoms
        '''
        self.cell_colors      = np.empty([8, 3], dtype=np.float32)

        for i, vertex in enumerate(vertices, 0):
        
            print("Vertex {}: {:7.3f} {:7.3f} {:7.3f}".format(i, vertex[0] ,vertex[1] ,vertex[2]))
            self.cell_indexes.append(i)
            
            self.cell_colors[i] = np.array(color, dtype=np.float32)
            self.cell_coordinates[0,i,:] = vertex[0] ,vertex[1] ,vertex[2]  
            

        print('\n\n')
        #print (self.cell_colors)







def find_connected_components(graph):
    """
    
    graph = topology (self.topology)
    
    eg: 

        self.topology = {0: [1, 2, 1], 1: [0, 0], 2: [0], 3: [6, 4, 5], 6: [3], 4: [3], 5: [3], 7: [8, 9], 8: [7], 9: [7]}
    
    
    
    This version of the function also takes a graph represented as a dictionary and 
    returns a list of connected components, where each connected component is a 
    list of nodes.

    The function starts by initializing an empty set called visited to keep track 
    of the nodes that have been visited and an empty list called components to 
    store the connected components.

    The function then iterates over each node in the graph and checks if it has 
    been visited yet. If the node has not been visited, the function starts a DFS 
    from that node.

    The DFS is implemented using a stack-based approach. The function initializes 
    a stack with the starting node and an empty list called component to store the 
    nodes in the connected component.

    The function then enters a loop that continues as long as the stack is not 
    empty. In each iteration of the loop, the function pops a node from the stack 
    and checks if it has been visited yet. If the node has not been visited, the 
    function adds it to the visited set, appends it to the component list, and 
    adds its unvisited neighbors to the stack.

    After the DFS has completed, the function appends the component list to the 
    components list.

    Finally, the function returns the components list, which contains the connected 
    components of the graph. Each connected component is represented as a list of 
    nodes.
    
    eg:
        components = [[0, 1, 2], [3, 5, 4, 6], [7, 9, 8]]
    
    """
    
    
    visited = set()
    components = []

    for start_node in graph:
        if start_node not in visited:
            stack = [start_node]
            component = []
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.append(node)
                    stack.extend([neighbor for neighbor in graph[node] if neighbor not in visited])
            components.append(component)
    return components

def DFS(graph, node, visited):
    ''' 
        The DFS function takes a graph, a node, and a set of 
        visited nodes as inputs, and performs a depth-first 
        search starting from the node. 
        
        is not being used

    '''
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            DFS(graph, neighbor, visited)

def find_groups(graph):
    '''
    
    is not being used due: 
    
    Python: maximum recursion depth exceeded while calling a Python object

    for more info access: 
    https://stackoverflow.com/questions/6809402/python-maximum-recursion-depth-exceeded-while-calling-a-python-object
        
    '''
    visited = set()
    groups = []
    for node in graph:
        if node not in visited:
            group = set()
            DFS(graph, node, group)
            groups.append(group)
            visited |= group
    return groups
