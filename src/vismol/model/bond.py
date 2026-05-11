#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  bond.py
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

import numpy as np
import math

class Bond:
    """ Class doc """
    
    def __init__(self, atom_i, atom_j, atom_index_i=None, atom_index_j=None, bond_order=1):
        """ Class initialiser """
        self.atom_i = atom_i
        self.atom_j = atom_j
        # - Remember that the "index" attribute refers to the numbering of atoms 
        # (it is not a zero base, it starts at 1 for the first atom)
        # these indices are zero base numbering (it starts at 0 for the first atom)
        if atom_index_i:
            self.atom_index_i = atom_index_i
        else:
            self.atom_index_i = atom_i.index-1
        
        if atom_index_j:
            self.atom_index_j = atom_index_j
        else:
            self.atom_index_j = atom_j.index-1
        
        self.bond_order = bond_order
        self.line_active = True
        self.stick_active = False
    
        self.bond_reference = None


    def get_bond_order(self):
        """Determina a ordem de ligação com base na distância."""
        
        if self.bond_reference is None:
            self.get_bond_reference()
        
        dist = self.distance()
        
        # ordenar por comprimento (menor = maior ordem)
        refs = sorted(self.bond_reference.items(), key=lambda x: x[1])
        
        self.bond_order = 1  # default
        
        for order, ref_dist in refs:
            if dist <= ref_dist:
                self.bond_order = order
                break

        print(self.atom_i.name, self.atom_j.name, dist, self.bond_order, self.bond_reference[self.bond_order])
        
        
        
        
    def get_bond_reference(self):
        """
        Calcula comprimentos de ligação r_ij para diferentes ordens
        usando o modelo do UFF.
        """
        if self.bond_reference is None:
            _lambda = 0.1332

            ri = self.atom_i.cov_rad
            rj = self.atom_j.cov_rad

            Xi = self.atom_i.electronegativity
            Xj = self.atom_j.electronegativity

            self.bond_reference = {}

            bond_orders = [1, 1.5, 2, 3]

            for n in bond_orders:
                rBO = -_lambda * (ri + rj) * math.log(n)

                rEN = (ri * rj * (math.sqrt(Xi) - math.sqrt(Xj))**2) / (Xi * ri + Xj * rj)

                r_ij = ri + rj + rBO + rEN

                self.bond_reference[n] = r_ij
        

        
        #print ( self.atom_i.name ) 
        #print ( self.atom_j.name )
        #print ( self.bond_reference)
        #print ( self.distance ( ))
        #if 
        
        #return self.bond_reference

    #def get_r_BO (self):
    #    """ Function doc 
    #    
    #    Xi, Xj eletronegatividades (escala de Eletronegatividade de Pauling - do UFF)
    #    
    #    r_ij =r_i + r_j + rBO +rEN
    #    
    #    """
    #    
    #    _lambda = 0.1332
    #    
    #    ri = self.atom_i.cov_rad
    #    rj​ = self.atom_j.cov_rad
    #    
    #    Xi = self.atom_i.electronegativity
    #    Xj = self.atom_j.electronegativity
    #    
    #    for n in range(1, 4):
    #       
    #        rBO ​= −_lambda(ri​ + rj​) * math.log(n)
    #        rEN = ri​*rj​*(math.sqrt(Xi) - math.sqrt(Xj))**2 / (Xi*ri​ + Xj*rj)
    #        r_ij =ri + rj + rBO +rEN
    #        self.bond_reference[n] = r_ij
    #    print(self.bond_reference)
    
    
    def distance (self, frame=0):
        """ Function doc """
        vec = self.atom_i.coords(frame) - self.atom_j.coords(frame)
        return np.linalg.norm(vec)
