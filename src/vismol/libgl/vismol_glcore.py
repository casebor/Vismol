#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  vismol_glcore.py
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
from OpenGL import GL
from logging import getLogger
from vismol.libgl.glaxis import GLAxis
from vismol.libgl.glcamera import GLCamera
from vismol.libgl.vismol_font import VismolFont
from vismol.libgl.selection_box import SelectionBox
import vismol.libgl.shapes as shapes
import vismol.libgl.shaders.pick as shaders_pick
import vismol.libgl.shaders.dots as shaders_dots
import vismol.libgl.shaders.lines as shaders_lines
import vismol.libgl.shaders.wires as shaders_wires
import vismol.libgl.shaders.sticks as shaders_sticks
import vismol.libgl.shaders.cartoon as shaders_cartoon
import vismol.libgl.shaders.surface as shaders_surface
import vismol.libgl.shaders.spheres as shaders_spheres
import vismol.libgl.shaders.impostor as shaders_impostor
import vismol.libgl.shaders.nonbonded as shaders_nonbonded
import vismol.libgl.shaders.vm_freetype as shaders_vm_freetype
import vismol.libgl.shaders.dashed_lines as shaders_dashed_lines
import vismol.utils.matrix_operations as mop

from vismol.core.vismol_object import VismolObject
from vismol.model.atom import Atom
import math
logger = getLogger(__name__)


class VismolGLCore:
    
    def __init__(self, widget, vismol_session=None, width=640.0, height=420.0):
        """ Constructor of the class.
            
            Keyword arguments:
            vismol_session - 
        """
        self.parent_widget = widget
        self.vm_session = vismol_session
        self.vm_config = self.vm_session.vm_config
        self.width = np.float32(width)
        self.height = np.float32(height)
        self.shader_programs = {}
        self.core_shader_programs = {}
        self.representations_available = self.vm_config.representations_available
        self.sphere_selection = None
        # Cache for glGetUniformLocation results, keyed by (program, name).
        # Uniform locations are fixed once a program is linked, so looking
        # them up by string every frame (the previous behaviour) is wasted
        # driver/CPU work. This cache is invalidated whenever shaders are
        # (re)compiled - see create_gl_programs().
        self._uniform_loc_cache = {}
        # --- Medidor de frame-time (Gargalo: instrumentacao) ---------------
        # Leve e opcional. Quando self.show_fps == True, render() acumula o
        # tempo de cada frame e imprime FPS medio + ms/frame a cada
        # self._fps_report_every frames. Default desligado: nenhum custo alem
        # de um 'if' por frame quando False. Para medir, basta setar
        # glcore.show_fps = True em runtime (ou aqui). Totalmente reversivel.
        self.show_fps = False
        self._fps_report_every = 60      # reporta a cada N frames
        self._fps_frame_count = 0        # frames desde o ultimo report
        self._fps_accum_time = 0.0       # tempo acumulado (s) desde o report
        self._fps_last_t = None          # timestamp do frame anterior
        # -------------------------------------------------------------------
        
        
    def initialize(self):
        """ Enables the buffers and other charasteristics of the OpenGL context.
            sets the initial projection, view and model matrices
            
            self.flag -- Needed to only create one OpenGL program, otherwise a bunch of
                         programs will be created and use system resources. If the OpenGL
                         program will be changed change this value to True
        """
        self.model_mat = np.identity(4, dtype=np.float32) # Not sure if this is used :S
        # --- UBO de matrizes de camera (Gargalo 2) -------------------------
        # view_mat e proj_mat sao IGUAIS para todos os objetos do frame, mas
        # antes eram re-enviados por objeto via glUniformMatrix4fv (2 uploads
        # x N objetos por frame). Agora vivem num Uniform Buffer Object unico,
        # atualizado 1x por frame e compartilhado por todos os shaders via o
        # binding point CAMERA_UBO_BINDING. model_mat segue como uniform
        # por-objeto (muda por objeto, nao entra aqui).
        # Layout std140: dois mat4 contiguos = 2 * 64 = 128 bytes.
        #   offset   0: mat4 view_mat
        #   offset  64: mat4 proj_mat
        self.CAMERA_UBO_BINDING = 0          # binding point fixo
        self._camera_ubo = None              # id do buffer GL (lazy init)
        self._camera_ubo_size = 128          # 2 mat4 float32
        # -------------------------------------------------------------------
        self.zero_reference_point = np.zeros(3, dtype=np.float32)
        self.glcamera = GLCamera(self.vm_config.gl_parameters["field_of_view"],
                                 self.width / self.height,
                                 np.array([0,0,10], dtype=np.float32),
                                 self.zero_reference_point)
        
        self.vm_font        = VismolFont(color=[1, 1, 1, 1])
        self.vm_font_static = VismolFont(color=[1, 1, 1, 1])
        self.vm_font_dist   = VismolFont(char_res=264,char_width=0.17, char_height=0.17, color = [1, 1, 1, 1])
        
        self.axis = GLAxis(vm_glcore = self)
        self.selection_box = SelectionBox()
        self.parent_widget.set_has_depth_buffer(True)
        self.parent_widget.set_has_alpha(True)
        self.scroll = self.vm_config.gl_parameters["scroll_step"]
        self.bckgrnd_color = self.vm_config.gl_parameters["background_color"]
        #                       Light Parameters                                
        self.light_position = self.vm_config.gl_parameters["light_position"]
        self.light_ambient_coef = self.vm_config.gl_parameters["light_ambient_coef"]
        self.light_shininess = self.vm_config.gl_parameters["light_shininess"]
        self.light_intensity = self.vm_config.gl_parameters["light_intensity"]
        #                              Variables                                
        self.right = self.width / self.height
        self.left = -self.right
        self.top = np.float32(1.0)
        self.bottom = np.float32(-1.0)
        self.button = None
        self.dist_cam_zrp = np.linalg.norm(self.glcamera.get_position())
        self.shader_flag = True
        self.modified_data = False # Used somewhere?
        self.updated_coords = False
        self.dragging = False
        self.editing_mols = False
        self.show_axis = True
        self.ctrl = False
        self.shift = False
        self.atom_picked = None
        self.selection_box_picking = False
        self.picking = False
        self.picking_x, self.picking_y = None, None
        self.show_selection_box = False
        self.show_selection_box_x, self.show_selection_box_y = None, None
        self.mouse_x, self.mouse_y = np.float32(0.0), np.float32(0.0)
        self.mouse_rotate, self.mouse_zoom, self.mouse_pan = False, False, False
        self.drag_pos_x, self.drag_pos_y, self.drag_pos_z = None, None, None
    
    def resize_window(self, width, height):
        """ Resizing function, takes the widht and height of the widget
            and modifies the view in the camera acording to the new values
        
            Keyword arguments:
            width -- Actual width of the window
            height -- Actual height of the window
        """
        self.width = np.float32(width)
        self.height = np.float32(height)
        self.right = self.width / self.height
        self.left = -self.right
        self.center_x = self.width / 2.0
        self.center_y = self.height / 2.0
        self.glcamera.viewport_aspect_ratio = self.width / self.height
        _proj_mat = mop.my_glPerspectivef(self.glcamera.field_of_view,
                                          self.glcamera.viewport_aspect_ratio,
                                          self.glcamera.z_near, self.glcamera.z_far)
        self.glcamera.set_projection_matrix(_proj_mat)
        
    def mouse_pressed(self, button_number, mouse_x, mouse_y):
        """ Function doc
        """
        left   = np.int32(button_number) == 1
        middle = np.int32(button_number) == 2
        right  = np.int32(button_number) == 3
        self.mouse_rotate = left   and not (middle or right)
        self.mouse_zoom   = right  and not (middle or left)
        self.mouse_pan    = middle and not (right  or left)
        self.mouse_x = np.float32(mouse_x)
        self.mouse_y = np.float32(mouse_y)
        self.drag_pos_x, self.drag_pos_y, self.drag_pos_z = self._mouse_pos(self.mouse_x, self.mouse_y)
        self.dragging = False
        if left:
            
            if  self.ctrl:
                pass
                #print(self.shift, self.ctrl)
            
            if self.shift and not self.ctrl:
                #print('picking_selection_mode:',self.vm_session.picking_selection_mode)
                #
                if self.vm_session.picking_selection_mode: # bachega 06 / 18 /2025
                    #print(mouse_x, mouse_y, button_number)
                    pass 
                else: # bachega 06 / 18 /2025
                    self.show_selection_box = True
                    self.selection_box.start = self.get_viewport_pos(mouse_x, mouse_y)
                    self.selection_box.end = self.get_viewport_pos(mouse_x, mouse_y)
                    self.selection_box.update_points()
                    self.selection_box_x = mouse_x
                    self.selection_box_y = self.height - mouse_y

        if middle:
            self.picking_x = np.float32(mouse_x)
            self.picking_y = np.float32(mouse_y)
            self.picking = True
            self.parent_widget.queue_draw()
        if right:
            self.picking_x = np.float32(mouse_x)
            self.picking_y = np.float32(mouse_y)
            self.picking = True
            self.parent_widget.queue_draw()
    
    def mouse_released(self, button_number, mouse_x, mouse_y):
        """ Function doc
        int(event.button)
        
        info      = menu header info
        menu_type = "pick_menu" / "bg_menu" / "sele_menu" / "ob_menu"
        
        """
        left   = np.int32(button_number) == 1
        middle = np.int32(button_number) == 2
        right  = np.int32(button_number) == 3
        self.mouse_rotate = False
        self.mouse_zoom = False
        self.mouse_pan = False
        if self.dragging:
            if left:
                if self.shift:
                    self.selection_box_picking = True
                    self.show_selection_box = False
                    self.selection_box.start = None
                    self.selection_box.end = None
                    self.parent_widget.queue_draw()
        else:
            if left:
                self.picking_x = np.float32(mouse_x)
                self.picking_y = np.float32(mouse_y)
                self.picking = True
                self.button = 1
                #dragging is set to false here
                self.dragging = False
                self.parent_widget.queue_draw()
            if middle:
                if self.atom_picked is not None:
                    self.dragging = True
                    self.button = 2
                    self.center_on_atom(self.atom_picked)
                    self.atom_picked = None
            if right:
                # The right button (button = 3) always opens one of the available menus.
                self.button = 3
                menu_type = None
                # Check if there is anything in the selection list
                # If {} means that there are no selection points on the screen
                # Checks if vismol_session.current_selection has any selection.
                # Also needs to check whether "picking" mode is enabled.
                if not bool(self.vm_session.selections[self.vm_session.current_selection].selected_objects) \
                   or self.vm_session.picking_selection_mode:
                    # Checks if the list of atoms selected by the picking function has any elements. 
                    # If the list is empty, the pick menu is not shown.
                    if self.vm_session.picking_selection_mode \
                       and self.vm_session.picking_selections.picking_selections_list != [None,None,None,None]:
                        info = None
                        menu_type = "pick_menu"
                    else:
                        # Here the obj_menu is activated based on the atom that
                        # was identified by the picking function.
                        # The picking function detects the selected pixel and
                        # associates it with the respective object.
                        # There is no selection (blue dots) but an atom was
                        # identified in the click with the right button
                        if self.atom_picked is not None:
                            # Getting the info about the atom that was identified in the click
                            label = "{} / {} / {}({}) / {}({} / {})".format(self.atom_picked.vm_object.name,
                                                                        self.atom_picked.chain.name,
                                                                        self.atom_picked.residue.name,
                                                                        self.atom_picked.residue.index,
                                                                        self.atom_picked.name,
                                                                        self.atom_picked.index,
                                                                        self.atom_picked.symbol)
                            self.info_atom = self.atom_picked
                            self.atom_picked = None
                            menu_type = "obj_menu"
                            info = label
                        else:
                            # When no atom is identified in the click (user clicked
                            # on a point in the background)
                            menu_type = "bg_menu"
                            info = None
                else:
                    # When a selection (viewing selection) is active, the
                    # selection menu is passed as an option.
                    info = self.vm_session.selections[self.vm_session.current_selection].get_selection_info()
                    menu_type = "sele_menu"
                # The right button (button = 3) always opens one of the available menus.
                self.parent_widget.show_gl_menu(menu_type=menu_type, info=info)
        
        
        
        self.dragging = False
        self.parent_widget.queue_draw()
    
    def mouse_motion(self, mouse_x, mouse_y):
        """ Function doc
        """
        x = np.float32(mouse_x)
        y = np.float32(mouse_y)
        dx = x - self.mouse_x
        dy = y - self.mouse_y
        if (dx == 0) and (dy == 0):
            return
        self.mouse_x, self.mouse_y = x, y
        changed = False
        if self.mouse_rotate:
            changed = self._rotate_view(dx, dy, x, y)
        elif self.mouse_pan:
            changed = self._pan_view(x, y)
        elif self.mouse_zoom:
            changed = self._zoom_view(dy)
        if changed:
            self.dragging = True
            self.parent_widget.queue_draw()
        
        #else:
        #    self.dragging = False
        #    self.parent_widget.queue_draw()
            
    
    def mouse_scroll(self, direction):
        """ Function doc
        """
        up = np.int32(direction) == 1
        down = np.int32(direction) == -1
        if self.ctrl:
            if self.editing_mols:
                for index, vm_object in self.vm_session.vm_objects_dic.items():
                    if vm_object.editing:
                        if up:
                            vm_object.model_mat = mop.my_glTranslatef(vm_object.model_mat, np.array([0.0, 0.0, -self.scroll]))
                        if down:
                            vm_object.model_mat = mop.my_glTranslatef(vm_object.model_mat, np.array([0.0, 0.0, self.scroll]))
                
                for key in self.vm_session.vm_geometric_object_dic.keys():
                    vm_object = self.vm_session.vm_geometric_object_dic[key]
                    if vm_object:
                        if vm_object.editing:
                            if up:
                                vm_object.model_mat = mop.my_glTranslatef(vm_object.model_mat, np.array([0.0, 0.0, -self.scroll]))
                            if down:
                                vm_object.model_mat = mop.my_glTranslatef(vm_object.model_mat, np.array([0.0, 0.0, self.scroll]))
            else:
                if up:
                    self.model_mat = mop.my_glTranslatef(self.model_mat, np.array([0.0, 0.0, -self.scroll]))
                if down:
                    self.model_mat = mop.my_glTranslatef(self.model_mat, np.array([0.0, 0.0, self.scroll]))
                
                for index, vm_object in self.vm_session.vm_objects_dic.items():
                    if up:
                        vm_object.model_mat = mop.my_glTranslatef(vm_object.model_mat, np.array([0.0, 0.0, -self.scroll]))
                    if down:
                        vm_object.model_mat = mop.my_glTranslatef(vm_object.model_mat, np.array([0.0, 0.0, self.scroll]))
                
                for key in self.vm_session.vm_geometric_object_dic.keys():
                    vm_object = self.vm_session.vm_geometric_object_dic[key]
                    if vm_object:
                        if up:
                            vm_object.model_mat = mop.my_glTranslatef(vm_object.model_mat, np.array([0.0, 0.0, -self.scroll]))
                        if down:
                            vm_object.model_mat = mop.my_glTranslatef(vm_object.model_mat, np.array([0.0, 0.0, self.scroll]))
        
        elif self.shift:
            #print(direction)
            #theta = direction/10
            
            # Editing Dihedral Rotamer / moving group os atoms
            if self.vm_session.picking_selection_mode:
                theta = direction/10
                atom1 = self.vm_session.picking_selections.picking_selections_list[0]
                atom2 = self.vm_session.picking_selections.picking_selections_list[1]
                atom3 = self.vm_session.picking_selections.picking_selections_list[2]
                atom4 = self.vm_session.picking_selections.picking_selections_list[3]
                
                if atom1 and atom2 and atom3 and atom4:
                                       
                    subgroup = self.vm_session.picking_selections.subgroup
                    index1   = atom2.index-1
                    index2   = atom3.index-1
                    vobject  = atom2.vm_object
                    #print(subgroup)
                    #self.vm_session.get_dihedral (theta)
                    
                    self.vm_session.rotate_dihedral ( 
                                       vobject   = vobject, 
                                       index1    = index1, 
                                       index2    = index2, 
                                       subgroup  = subgroup,
                                       theta     = theta)
                
                elif atom1 and atom2 and not atom3 and not atom4:
                    #print (atom1 ,atom2)
                    index1   = atom1.index-1
                    index2   = atom2.index-1
                    subgroup = self.vm_session.picking_selections.subgroup
                    vobject  = atom2.vm_object
                    #print(direction)
                    self.vm_session.move_subgroup  (
                                       vobject   = vobject,
                                       index1    = index1, 
                                       index2    = index2, 
                                       subgroup  = subgroup,
                                       direction = direction)
                                       
                    
                    
                    
        else:
            pos_z = self.glcamera.get_position()[2]
            if up:
                self.glcamera.z_near -= self.scroll
                self.glcamera.z_far += self.scroll
            if down:
                if (self.glcamera.z_far-self.scroll) >= (self.glcamera.min_zfar):
                    if (self.glcamera.z_far-self.scroll) > (self.glcamera.z_near + self.scroll + 0.005):
                        self.glcamera.z_near += self.scroll
                        self.glcamera.z_far -= self.scroll
            
            if (self.glcamera.z_near >= self.glcamera.min_znear):
                self.glcamera.set_projection_matrix(mop.my_glPerspectivef(self.glcamera.field_of_view,
                        self.glcamera.viewport_aspect_ratio, self.glcamera.z_near, self.glcamera.z_far))
            else:
                if self.glcamera.z_far < (self.glcamera.min_zfar + self.glcamera.min_znear):
                    self.glcamera.z_near -= self.scroll
                    self.glcamera.z_far = self.glcamera.min_clip + self.glcamera.min_znear
                self.glcamera.set_projection_matrix(mop.my_glPerspectivef(self.glcamera.field_of_view,
                                                    self.glcamera.viewport_aspect_ratio,
                                                    self.glcamera.min_znear, self.glcamera.z_far))
            self.glcamera.update_fog()
        self.parent_widget.queue_draw()

    def _rotate_view(self, dx, dy, x, y):
        """
        Rotate the scene or selected objects based on mouse movement.
        """

        sens = self.vm_config.gl_parameters['mouse_rotation_sensibility']
        dx *= sens
        dy *= sens

        angle = np.hypot(dx, dy) / max(self.width, self.height) * 180.0

        # SHIFT → selection box update
        if self.shift:
            if self.vm_session.picking_selection_mode:
                return False

            self.selection_box.end = self.get_viewport_pos(self.mouse_x, self.mouse_y)
            self.selection_box.update_points()
            return True

        # --- Compute rotation axis ---
        if self.ctrl:
            if abs(dx) >= abs(dy):
                sign = -1 if (y - self.height / 2.0) >= 0 else 1
                axis = np.array([0.0, 0.0, sign * dx], dtype=np.float32)
            else:
                sign = -1 if (x - self.width / 2.0) >= 0 else 1
                axis = np.array([0.0, 0.0, sign * dy], dtype=np.float32)
        else:
            axis = np.array([-dy, -dx, 0.0], dtype=np.float32)

        # Normalize axis
        norm = np.linalg.norm(axis)
        if norm == 0:
            return False
        axis /= norm

        rotation_matrix = mop.my_glRotatef(np.identity(4), angle, axis)

        # --- Apply rotation ---
        def iter_all_objects():
            yield from self.vm_session.vm_objects_dic.values()
            for obj in self.vm_session.vm_geometric_object_dic.values():
                if obj:
                    yield obj

        if self.editing_mols:
            for obj in iter_all_objects():
                if obj.editing:
                    obj.model_mat = mop.my_glMultiplyMatricesf(obj.model_mat, rotation_matrix)
        else:
            self.model_mat = mop.my_glMultiplyMatricesf(self.model_mat, rotation_matrix)
            for obj in iter_all_objects():
                obj.model_mat = mop.my_glMultiplyMatricesf(obj.model_mat, rotation_matrix)

        # --- Gizmo axis ---
        if not self.editing_mols:
            self.axis.model_mat = mop.my_glTranslatef(self.axis.model_mat, -self.axis.zrp)
            self.axis.model_mat = mop.my_glRotatef(self.axis.model_mat, angle, axis)
            self.axis.model_mat = mop.my_glTranslatef(self.axis.model_mat, self.axis.zrp)

        return True                 

    def _rotate_view_old(self, dx, dy, x, y):
        """ Function doc """
        dx =  dx*self.vm_config.gl_parameters['mouse_rotation_sensibility']
        dy =  dy*self.vm_config.gl_parameters['mouse_rotation_sensibility']
        angle = np.sqrt(dx**2 + dy**2) / (self.width + 1) * 180.0
        if self.shift:
            if self.vm_session.picking_selection_mode:
                return False
            else:
                self.selection_box.end = self.get_viewport_pos(self.mouse_x, self.mouse_y)
                self.selection_box.update_points()
        
        else:
            if self.ctrl:
                if abs(dx) >= abs(dy):
                    if (y - self.height / 2.0) < 0:
                        rot_mat = mop.my_glRotatef(np.identity(4), angle, np.array([0.0, 0.0, dx]))
                    else:
                        rot_mat = mop.my_glRotatef(np.identity(4), angle, np.array([0.0, 0.0, -dx]))
                else:
                    if (x - self.width / 2.0) < 0:
                        rot_mat = mop.my_glRotatef(np.identity(4), angle, np.array([0.0, 0.0, -dy]))
                    else:
                        rot_mat = mop.my_glRotatef(np.identity(4), angle, np.array([0.0, 0.0, dy]))
            else:
                rot_mat = mop.my_glRotatef(np.identity(4), angle, np.array([-dy, -dx, 0.0]))
            
            if self.editing_mols:
                for index, vm_object in self.vm_session.vm_objects_dic.items():
                    if vm_object.editing:
                        vm_object.model_mat = mop.my_glMultiplyMatricesf(vm_object.model_mat, rot_mat)
                
                for key in self.vm_session.vm_geometric_object_dic.keys():
                    vm_object = self.vm_session.vm_geometric_object_dic[key]
                    if vm_object:
                        if vm_object.editing:
                            vm_object.model_mat = mop.my_glMultiplyMatricesf(vm_object.model_mat, rot_mat)
            else:
                self.model_mat = mop.my_glMultiplyMatricesf(self.model_mat, rot_mat)
                for index, vm_object in self.vm_session.vm_objects_dic.items():
                    vm_object.model_mat = mop.my_glMultiplyMatricesf(vm_object.model_mat, rot_mat)
                
                for key in self.vm_session.vm_geometric_object_dic.keys():
                    vm_object = self.vm_session.vm_geometric_object_dic[key]
                    if vm_object:
                        vm_object.model_mat = mop.my_glMultiplyMatricesf(vm_object.model_mat, rot_mat)
            
            # Axis operations, this code only affects the gizmo axis
            if not self.editing_mols:
                self.axis.model_mat = mop.my_glTranslatef(self.axis.model_mat, -self.axis.zrp)
                if self.ctrl:
                    if abs(dx) >= abs(dy):
                        if (y - self.height / 2.0) < 0:
                            self.axis.model_mat = mop.my_glRotatef(self.axis.model_mat, angle, np.array([0.0, 0.0, dx]))
                        else:
                            self.axis.model_mat = mop.my_glRotatef(self.axis.model_mat, angle, np.array([0.0, 0.0, -dx]))
                    else:
                        if (x - self.width / 2.0) < 0:
                            self.axis.model_mat = mop.my_glRotatef(self.axis.model_mat, angle, np.array([0.0, 0.0, -dy]))
                        else:
                            self.axis.model_mat = mop.my_glRotatef(self.axis.model_mat, angle, np.array([0.0, 0.0, dy]))
                else:
                    self.axis.model_mat = mop.my_glRotatef(self.axis.model_mat, angle, np.array([dy, dx, 0.0]))
                self.axis.model_mat = mop.my_glTranslatef(self.axis.model_mat, self.axis.zrp)
                #self.axis.model_mat = mop.my_glScalef(self.axis.model_mat, self.axis.zrp)
                
            # Axis operations, this code only affects the gizmo axis
        return True
    
    def _pan_view(self, x, y):
        """ Function doc """
        px, py, pz = self._mouse_pos(x, y)
        pan_mat = mop.my_glTranslatef(np.identity(4, dtype=np.float32), np.array(
                                    [(px - self.drag_pos_x) * self.glcamera.z_far / 10.0,
                                     (py - self.drag_pos_y) * self.glcamera.z_far / 10.0,
                                     (pz - self.drag_pos_z) * self.glcamera.z_far / 10.0]))
        if self.editing_mols:
            for index, vm_object in self.vm_session.vm_objects_dic.items():
                if vm_object.editing:
                    vm_object.model_mat = mop.my_glMultiplyMatricesf(vm_object.model_mat, pan_mat)
            
            for key in self.vm_session.vm_geometric_object_dic.keys():
                vm_object = self.vm_session.vm_geometric_object_dic[key]
                if vm_object:
                    if vm_object.editing:
                        vm_object.model_mat = mop.my_glMultiplyMatricesf(vm_object.model_mat, pan_mat)
        
        else:
            self.model_mat = mop.my_glMultiplyMatricesf(self.model_mat, pan_mat)
            for index, vm_object in self.vm_session.vm_objects_dic.items():
                vm_object.model_mat = mop.my_glMultiplyMatricesf(vm_object.model_mat, pan_mat)
            
            for key in self.vm_session.vm_geometric_object_dic.keys():
                vm_object = self.vm_session.vm_geometric_object_dic[key]
                if vm_object:
                    vm_object.model_mat = mop.my_glMultiplyMatricesf(vm_object.model_mat, pan_mat)
            self.zero_reference_point = mop.get_xyz_coords(self.model_mat)
        
        self.drag_pos_x = px
        self.drag_pos_y = py
        self.drag_pos_z = pz
        return True
    
    def _zoom_view(self, dy):
        """ Function doc """
        delta = (((self.glcamera.z_far - self.glcamera.z_near) / 2.0) + self.glcamera.z_near) / 200.0
        move_z = dy * delta
        moved_mat = mop.my_glTranslatef(self.glcamera.view_matrix, np.array([0.0, 0.0, move_z]))
        moved_pos = mop.get_xyz_coords(moved_mat)
        if moved_pos[2] > 0.101:
            self.glcamera.set_view_matrix(moved_mat)
            self.glcamera.z_near -= move_z
            self.glcamera.z_far -= move_z
            if self.glcamera.z_near >= self.glcamera.min_znear:
                self.glcamera.set_projection_matrix(mop.my_glPerspectivef(self.glcamera.field_of_view, 
                                                    self.glcamera.viewport_aspect_ratio,
                                                    self.glcamera.z_near, self.glcamera.z_far))
            else:
                if self.glcamera.z_far < (self.glcamera.min_zfar+self.glcamera.min_znear):
                    self.glcamera.z_near += move_z
                    self.glcamera.z_far = self.glcamera.min_zfar+self.glcamera.min_znear
                self.glcamera.set_projection_matrix(mop.my_glPerspectivef(self.glcamera.field_of_view, 
                                                    self.glcamera.viewport_aspect_ratio,
                                                    self.glcamera.min_znear, self.glcamera.z_far))
            self.glcamera.update_fog()
            self.dist_cam_zrp += -move_z
            return True
        return False
    
    def render(self, rotating = False):
        """ This is the function that will be called everytime the window
            needs to be re-drawed.
        """
        # Medidor opcional: marca o inicio do corpo do render. Mede o custo
        # de CPU+submissao GL deste frame (nao inclui o tempo ocioso entre
        # frames), que e exatamente o numero relevante para avaliar se o
        # refactor de UBO (Gargalo 2) compensa.
        if self.show_fps:
            _t_start = time.perf_counter()
        if self.shader_flag:
            self.create_gl_programs()
            self.selection_box.initialize_gl()
            self.axis.initialize_gl()
            self.shader_flag = False
        if self.selection_box_picking:
            self._selection_box_pick()
        if self.picking:
            self._pick()
        
        #print('self.dragging', self.dragging)
        
        GL.glClearColor(self.bckgrnd_color[0], self.bckgrnd_color[1],
                        self.bckgrnd_color[2], self.bckgrnd_color[3])
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # Atualiza view/proj no UBO uma unica vez por frame (Gargalo 2).
        # Todos os shaders convertidos leem deste buffer compartilhado.
        self.update_camera_ubo()
        
        if self.updated_coords:
            for vm_object in self.vm_session.vm_objects_dic.values():
                if vm_object.core_representations["picking_dots"] is None:
                    vm_object.build_core_representations()
                vm_object.core_representations["picking_dots"].was_rep_coord_modified = True
                for rep in vm_object.representations.values():
                    if rep is not None:
                        # Only flag coordinates as dirty for representations
                        # that are actually drawn. Inactive representations
                        # (e.g. a hidden "spheres" rep) would otherwise have
                        # their full coordinate VBO re-uploaded to the GPU
                        # every single frame of trajectory/MD playback for
                        # nothing, since draw_representation() never runs
                        # for them while inactive. was_sel_coord_modified is
                        # safe to skip here too: draw_background_sel_representation()
                        # (which consumes that flag) is itself only called
                        # for representations where rep.active is True - see
                        # _selection_box_pick(). Re-activating a representation
                        # already re-marks its indexes as modified elsewhere
                        # (vismol_session.py), and we mark coords there too
                        # below so a reactivated rep never draws stale coords.
                        if rep.active:
                            rep.was_rep_coord_modified = True
                            rep.was_sel_coord_modified = True
                        if rep.is_dynamic:
                            rep.was_rep_ind_modified = True
                            rep.was_sel_ind_modified = True
        
        '''
                           - - -  R E P R E S E N T A T I O N S - - -  
        '''
        for vm_object in self.vm_session.vm_objects_dic.values():
            if vm_object.active:
                if vm_object.frames.shape[0] > 0:
                    #print(vm_object.representations)
                    for representation in vm_object.representations.values():
                        if representation is not None:
                            # Only shows the representation if
                            # representations[rep_name].active = True
                            if representation.active:
                                representation.draw_representation()
                                
        
        if self.vm_session.picking_selection_mode:
            '''
                               - - -  P I C K I N G   S E L E C T I O N S - - -  
                                Here is where we will draw the dashed lines
                 Drawing labels on the fly is not very efficient; however, since there are only a few labels 
                 to be generated, the impact on performance is minimal.
            '''
            #'''
            '''Two loops are necessary so that the labels are always drawn above the dash lines.'''
            for geo_object in ["pk1pk2", "pk2pk3", "pk3pk4"]:
                if self.vm_session.vm_geometric_object_dic[geo_object]:
                    if self.vm_session.vm_geometric_object_dic[geo_object].representations["dash"].active:
                        self.vm_session.vm_geometric_object_dic[geo_object].representations["dash"].draw_representation()
            
            '''                              drawing distance labels                     '''
            for geo_object in ["pk1pk2", "pk2pk3", "pk3pk4"]:
                if self.vm_session.vm_geometric_object_dic[geo_object]:
                    if self.vm_session.vm_geometric_object_dic[geo_object].representations["dash"].active:
                        self._draw_distance_labels(self.vm_session.vm_geometric_object_dic[geo_object])
            
            '''                               drawing picking spheres                     '''
            for geo_object in ["pk1", "pk2", "pk3", "pk4"]:
                if self.vm_session.vm_geometric_object_dic[geo_object]:
                    if self.vm_session.vm_geometric_object_dic[geo_object].representations["picking_spheres"].active:
                        self.vm_session.vm_geometric_object_dic[geo_object].representations["picking_spheres"].draw_representation()
                        #print(geo_object,self.vm_session.vm_geometric_object_dic[geo_object].representations["picking_spheres"].active)
            '''                               drawing picking labels                     '''
            self._draw_picking_label()
            #------------------------------------------------------------------
            #'''
        
        
        else:
            '''                    - - -  V I E W I N G   S E L E C T I O N S - - -                  '''
            '''                               drawing blues dot selections                           '''
            for vm_object in self.vm_session.selections[self.vm_session.current_selection].selected_objects:
                # Here are represented the blue dots referring to the atom's selections
                if vm_object.core_representations["picking_dots"] is None:
                    vm_object.build_core_representations()
                pdots = vm_object.core_representations["picking_dots"]
                # Gargalo 1: a reconstrucao da lista de indices + re-upload do
                # VBO so precisa acontecer quando o conjunto de atomos
                # selecionados muda. Antes isso era feito INCONDICIONALMENTE
                # a cada frame (rebuild de lista Python, np.array novo e
                # glBufferData), desperdicando trabalho durante rotacao/zoom
                # com a selecao parada. Comparamos com um snapshot do ultimo
                # conjunto subido; so re-subimos no diff. O draw continua
                # todo frame, pois a cena e redesenhada normalmente.
                sel_ids = vm_object.selected_atom_ids
                if pdots._last_uploaded_sel_ids != sel_ids:
                    pdots.was_rep_ind_modified = True
                    pdots.define_new_indexes_to_vbo(list(sel_ids))
                    # snapshot por copia: selected_atom_ids e mutado in-place
                    # (add/discard/clear), entao guardar a referencia nao
                    # detectaria mudancas. set(...) congela o estado atual.
                    pdots._last_uploaded_sel_ids = set(sel_ids)
                pdots.draw_representation()
        
        if not self.vm_session.picking_selection_mode and self.show_selection_box and self.shift:
            if self.selection_box.vao is None:
                self.selection_box._make_gl_selection_box()
            else:
                self.selection_box._draw()
        
        if self.show_axis:
            self.axis._draw(True)
            self.axis._draw(False)

        # Fecha o cronometro do frame e reporta periodicamente. Medimos so o
        # tempo de CPU/submissao (sem glFinish, que serializaria a GPU e
        # distorceria o numero). 'ms/frame' aqui = custo da thread Python por
        # render; e o indicador para decidir sobre o UBO. O 'FPS' derivado e
        # o teto teorico se o render fosse o unico limite.
        if self.show_fps:
            dt = time.perf_counter() - _t_start
            self._fps_accum_time += dt
            self._fps_frame_count += 1
            if self._fps_frame_count >= self._fps_report_every:
                avg_s = self._fps_accum_time / self._fps_frame_count
                avg_ms = avg_s * 1000.0
                fps = (1.0 / avg_s) if avg_s > 0 else float("inf")
                n_obj = len(self.vm_session.vm_objects_dic)
                print("[FPS] {:.1f} fps | {:.3f} ms/render | {} objetos | "
                      "media de {} frames".format(fps, avg_ms, n_obj,
                                                   self._fps_frame_count))
                self._fps_frame_count = 0
                self._fps_accum_time = 0.0
        return True
    
    def render_to_image(self, scale_factor=1):
        """ Renders the current scene into an offscreen framebuffer at
            scale_factor times the widget's current resolution, then
            reads it back as a numpy RGBA array (height, width, 4) with
            origin at the top-left (already flipped from OpenGL's
            bottom-left convention, ready to hand to PIL/Image.fromarray).

            Rationale for *rendering* at a higher resolution instead of
            just upscaling the captured screenshot afterwards: naive
            image upscaling only blurs/interpolates existing pixels, it
            doesn't add detail. Rendering at scale_factor x resolution
            actually rasterizes thinner/more precise lines, smoother
            circles for spheres/dots, and crisper text, because the GPU
            recomputes every fragment at the higher pixel density.

            scale_factor must be uniform (same factor for width and
            height) so the aspect ratio - and therefore the projection
            matrix - stays exactly the same as the on-screen view; only
            the pixel density changes. This also means dot/point sizes
            (which several representations compute in screen pixels via
            vm_glcore.height - see representations.py) come out scaled
            consistently with line/stick thickness (which is computed in
            world units via the projection matrix), instead of looking
            disproportionately small at the higher resolution.

            Keyword arguments:
            scale_factor -- integer or float multiplier applied to both
                            width and height (e.g. 2 for 2x, 3 for 3x).

            Returns:
            image -- np.uint8 array of shape (height*scale_factor,
                     width*scale_factor, 4), or None if the offscreen
                     framebuffer could not be created/completed.
        """
        if scale_factor == 1:
            # No offscreen framebuffer needed - just read whatever is
            # already in the on-screen framebuffer (caller is expected
            # to have rendered a fresh frame into it already).
            width = int(self.width)
            height = int(self.height)
            data = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
            return np.ascontiguousarray(np.flip(image, axis=0))

        # --- Save every bit of state render_to_image is about to touch ---
        orig_width  = self.width
        orig_height = self.height
        orig_right  = self.right
        orig_left   = self.left
        orig_proj_mat = np.copy(self.glcamera.projection_matrix)
        orig_aspect   = self.glcamera.viewport_aspect_ratio
        prev_fbo      = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        prev_viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)

        new_width  = int(orig_width  * scale_factor)
        new_height = int(orig_height * scale_factor)

        fbo = color_tex = depth_rbo = None
        try:
            # --- Build the offscreen framebuffer at the higher resolution ---
            color_tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, color_tex)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, new_width, new_height,
                             0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

            depth_rbo = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_rbo)
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24,
                                      new_width, new_height)

            fbo = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                       GL.GL_TEXTURE_2D, color_tex, 0)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                          GL.GL_RENDERBUFFER, depth_rbo)

            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                logger.error("render_to_image: offscreen framebuffer incomplete (status=%s)", status)
                return None

            # --- Point width/height/projection at the higher resolution ---
            # Same aspect ratio (width and height scale by the same
            # factor), so the projection matrix comes out geometrically
            # identical to the on-screen one - this only changes pixel
            # density, not framing/zoom/distortion.
            self.width  = np.float32(new_width)
            self.height = np.float32(new_height)
            self.right  = self.width / self.height
            self.left   = -self.right
            self.glcamera.viewport_aspect_ratio = self.width / self.height
            self.glcamera.set_projection_matrix(
                mop.my_glPerspectivef(self.glcamera.field_of_view,
                                      self.glcamera.viewport_aspect_ratio,
                                      self.glcamera.z_near, self.glcamera.z_far))

            GL.glViewport(0, 0, new_width, new_height)
            self.render()

            data = GL.glReadPixels(0, 0, new_width, new_height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            image = np.frombuffer(data, dtype=np.uint8).reshape((new_height, new_width, 4))
            image = np.ascontiguousarray(np.flip(image, axis=0))
            return image

        finally:
            # --- Restore everything, even if something above raised ---
            self.width  = orig_width
            self.height = orig_height
            self.right  = orig_right
            self.left   = orig_left
            self.glcamera.viewport_aspect_ratio = orig_aspect
            self.glcamera.set_projection_matrix(orig_proj_mat)

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, prev_fbo)
            GL.glViewport(int(prev_viewport[0]), int(prev_viewport[1]),
                          int(prev_viewport[2]), int(prev_viewport[3]))

            if fbo is not None:
                GL.glDeleteFramebuffers(1, [fbo])
            if color_tex is not None:
                GL.glDeleteTextures(1, [color_tex])
            if depth_rbo is not None:
                GL.glDeleteRenderbuffers(1, [depth_rbo])

            # The widget's own framebuffer is now the wrong size for the
            # restored width/height until the next natural resize/render
            # cycle - request one so the on-screen view doesn't look
            # stretched if anything reads it before then.
            self.queue_draw()
    
    def _create_sphere_selection (self):
        """ Function doc """
        self.sphere_selection = VismolObject(self.vm_session, len(self.vm_session.vm_objects_dic), name='test')
        self.sphere_selection.set_model_matrix(self.model_mat)
        self.sphere_selection.create_representation(rep_type="picking_spheres")
        coords    = np.empty([1, 4, 3], dtype=np.float32)
        self.sphere_selection.frames = coords
        
        for index in range(0,4):
            atom = Atom(vismol_object = self.sphere_selection, 
                    name          ='Br'  , 
                    index         = index,
                    residue       = None , 
                    chain         = None, 
                    atom_id       = index,
                    occupancy     = 0, 
                    bfactor       = 0 ,
                    charge        = 0 )
            
            atom.vdw_rad  = 2.3 
            atom.cov_rad  = 2.3 
            atom.ball_rad = 2.3 
            color = [0.0,0.0,0.2]
            atom.color = np.array(color, dtype=np.float32)
            # unique_id and color_id (picking color) are None by default on
            # a freshly constructed Atom - they're normally set while
            # walking the parsed file in load_molecule(). This object is
            # built by hand, so set them explicitly; _generate_color_vectors
            # below needs a real atom.color_id to fill color_indexes.
            atom.unique_id = index
            atom._generate_atom_unique_color_id()
            self.sphere_selection.atoms[index] = atom
        
        # Build the object-level "colors"/"color_indexes" arrays from the
        # per-atom .color/.color_id values set above. Every normally loaded
        # VismolObject gets these via load_molecule() -> _generate_color_vectors();
        # this synthetic picking object skipped that step, so
        # SpheresRepresentation._colors_rads()/_sel_colors_rads() (which index
        # into vm_object.colors) would raise AttributeError as soon as
        # picking_spheres tried to draw.
        self.sphere_selection._generate_color_vectors(-1)
        
        self.sphere_selection.representations["picking_spheres"].define_new_indexes_to_vbo(range(0,4))
        self.sphere_selection.representations["picking_spheres"].active = True
        self.sphere_selection.representations["picking_spheres"].was_rep_ind_modified = True
        self.vm_session.vm_geometric_object_dic['picking_spheres'] = self.sphere_selection
        
        '''
        for index in range(1,5):
            atoms.append({"name"     : 'Au',
                          "resn"     : 'SEL',
                          "chain"    : 'A',
                          "resi"     : '1',
                          "occupancy": '1',
                          "bfactor"  : 0,
                          "charge"   : 0,
                          "index"    : index})
        
        for _atom in atoms:
            if _atom["chain"] not in vm_object.chains.keys():
                vm_object.chains[_atom["chain"]] = Chain(vm_object, name=_atom["chain"])
            _chain = vm_object.chains[_atom["chain"]]
            
            if _atom["resi"] not in _chain.residues.keys():
                _r = Residue(vm_object, name=_atom["resn"], index=_atom["resi"], chain=_chain)
                vm_object.residues[_atom["resi"]] = _r
                _chain.residues[_atom["resi"]] = _r
            _residue = _chain.residues[_atom["resi"]]
            
            atom = Atom(vm_object, name=_atom["name"], index=_atom["index"],
                        residue=_residue, chain=_chain, atom_id=atom_id,
                        occupancy=_atom["occupancy"], bfactor=_atom["bfactor"],
                        charge=_atom["charge"])
            atom.unique_id = unique_id
            atom._generate_atom_unique_color_id()
            _residue.atoms[atom_id] = atom
            vm_object.atoms[atom_id] = atom
            atom_id += 1
            unique_id += 1
        #logger.debug("Time used to build the tree: {:>8.5f} secs".format(time.time() - initial))
        vm_object.frames = PDBFiles.get_coords_from_raw_frames(rawframes, atom_id, vismol_session.vm_config.n_proc)
        '''

    def create_gl_programs(self):
        """ Function doc
        """
        logger.info("OpenGL version: {}".format(GL.glGetString(GL.GL_VERSION)))
        logger.info("OpenGL major version: {}".format(GL.glGetDoublev(GL.GL_MAJOR_VERSION)))
        logger.info("OpenGL minor version: {}".format(GL.glGetDoublev(GL.GL_MINOR_VERSION)))
        # Programs are about to be (re)linked, so any previously cached
        # uniform locations are no longer valid.
        self._uniform_loc_cache.clear()
        self._compile_shader_picking_dots()
        self._compile_shader_freetype()
        for rep in self.representations_available:
            func = getattr(self, "_compile_shader_" + rep)
            try:
                func()
            except AttributeError as ae:
                logger.error("Representation of type '{}' not implemented".format(rep))
                logger.error(ae)
    
    def load_shaders(self, vertex, fragment, geometry=None):
        """ Here the shaders are loaded and compiled to an OpenGL program. By default
            the constructor shaders will be used, if you want to change the shaders
            use this function. The flag is used to create only one OpenGL program.
            
            Keyword arguments:
            vertex -- The vertex shader to be used
            fragment -- The fragment shader to be used
        """
        my_vertex_shader = self.create_shader(vertex, GL.GL_VERTEX_SHADER)
        my_fragment_shader = self.create_shader(fragment, GL.GL_FRAGMENT_SHADER)
        if geometry is not None:
            my_geometry_shader = self.create_shader(geometry, GL.GL_GEOMETRY_SHADER)
        program = GL.glCreateProgram()
        GL.glAttachShader(program, my_vertex_shader)
        GL.glAttachShader(program, my_fragment_shader)
        if geometry is not None:
            GL.glAttachShader(program, my_geometry_shader)
        GL.glLinkProgram(program)
        # Liga o bloco de camera (se o shader o declara) ao binding point
        # compartilhado. No-op para shaders ainda nao convertidos.
        self._bind_camera_ubo_to_program(program)
        return program
    
    def create_shader(self, shader_prog, shader_type):
        """ Creates, links to a source, compiles and returns a shader.
            
            Keyword arguments:
            shader -- The shader text to use
            shader_type -- The OpenGL enum type of shader, it can be:
                           GL.GL_VERTEX_SHADER, GL.GL_GEOMETRY_SHADER or GL.GL_FRAGMENT_SHADER
            
            Returns:
            A shader object identifier or pops out an error
        """
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, shader_prog)
        GL.glCompileShader(shader)
        if GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
            logger.critical("Error compiling the shader: {}".format(shader_type))
            raise RuntimeError(logger.critical(GL.glGetShaderInfoLog(shader)))
        return shader

    def _selection_box_pick_new(self):
        """
        Select atoms using a rectangular selection box based on color picking.

        This method renders a special background where each atom is encoded as a unique color.
        Then, it reads pixels from the framebuffer inside the selection rectangle and decodes
        them back into atom IDs.
        """
        
        # Ensure mouse coordinates exist
        if not hasattr(self, "mouse_x") or not hasattr(self, "mouse_y"):
            return False
        
        # Original selection box corner (mouse press)
        x1, y1 = self.selection_box_x, self.selection_box_y
        # Current mouse position (convert Y from GTK to OpenGL coordinates)
        x2, y2 = self.mouse_x, self.height - self.mouse_y


        # Compute normalized rectangle (bottom-left corner + width/height)
        pos_x = int(max(0, min(x1, x2)))
        pos_y = int(max(0, min(y1, y2)))
        width = int(abs(x2 - x1))
        height = int(abs(y2 - y1))

        # Ignore empty selections
        if width == 0 or height == 0:
            return False

        # Clear buffers before rendering picking scene
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Render all active objects using their selection representation
        for vm_object in self.vm_session.vm_objects_dic.values():
            if not vm_object.active:
                continue
            for rep in vm_object.representations.values():
                if rep and rep.active:
                    rep.draw_background_sel_representation()

        # Ensure tight packing of pixel data (no row alignment padding)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        
        # Read pixels from framebuffer within the selection rectangle
        data = GL.glReadPixels(pos_x, pos_y, width, height,
                               GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)

        # Decode RGB values into unique IDs
        # Each pixel encodes an ID as: R + G*256 + B*256^2
        picked_set = {
            data[i] + data[i+1]*256 + data[i+2]*256*256
            for i in range(0, len(data), 4)
        }

        atom_dic = self.vm_session.atom_dic_id
        
        # Convert IDs into atom objects, ignoring background (white = 16777215)
        selected = {
            atom_dic[pid] for pid in picked_set
            if pid != 16777215 and pid in atom_dic
        }

        # Apply selection (disable=False means additive selection)
        self.vm_session._selection_function_set(selected, disable=False)
        
        # Disable selection box picking mode
        self.selection_box_picking = False

    def _selection_box_pick(self):
        """ Selects a set of atoms from pixels obtained by the rectangle selection.  
            This function (method) is called in the render method, when the 
            "self.selection_box_picking" attribute is active. 
        
        """
        
        # glReadPixels and glReadnPixels return pixel data from the frame buffer, 
        # starting with the pixel whose lower left corner is at location (x, y), 
        # into client memory starting at location data.
        #
        # In GTK, x=0 and y=0 set to upper left corner (unlike openGL input data, 
        # the following lines do the coordinate conversion) 
        
        #try: # bachega 06 / 18 /2025
        #    selection_box_x2 = self.mouse_x
        #    selection_box_y2 = self.height - self.mouse_y
        #    selection_box_width  = selection_box_x2 - self.selection_box_x
        #    selection_box_height = selection_box_y2 - self.selection_box_y
        #except: # bachega 06 / 18 /2025
        #    return False
        
        
        # Bachega 03/22/2026 
        mouse_x = getattr(self, "mouse_x", None)
        mouse_y = getattr(self, "mouse_y", None)

        if mouse_x is None or mouse_y is None:
            return False
        x2 = mouse_x
        y2 = self.height - mouse_y  # convert to OpenGL coordinates
        
        try:
            width = x2 - self.selection_box_x
            height = y2 - self.selection_box_y
        except:
            return None

        '''
        #Simplificar cálculo do retângulo (grande melhoria)
        #Looking for the lower left corner of the checkbox
        if selection_box_width > 0 and selection_box_height > 0:
            pos_x = self.selection_box_x
            pos_y = self.selection_box_y
            width = selection_box_width
            height = selection_box_height
        
        elif selection_box_width < 0 and selection_box_height > 0:
            pos_x = selection_box_x2
            pos_y = self.selection_box_y
            width = -selection_box_width
            height = selection_box_height
        
        elif selection_box_width < 0 and selection_box_height < 0:
            pos_x = selection_box_x2
            pos_y = selection_box_y2
            width =  -selection_box_width
            height = -selection_box_height
        else:
            pos_x = self.selection_box_x
            pos_y = selection_box_y2
            width = selection_box_width
            height = -selection_box_height
        
        # taking the module from the width and height values 
        if pos_x < 0:
            pos_x = 0.0
        if pos_y < 0:
            pos_y = 0.0
        #'''
        
        #       cleaner and shoter bachega 03/22/2026
        # -------------------------------------------------
        x1 = self.selection_box_x
        y1 = self.selection_box_y
        x2 = self.mouse_x
        y2 = self.height - self.mouse_y

        pos_x = int(min(x1, x2))
        pos_y = int(min(y1, y2))
        width = int(abs(x2 - x1))
        height = int(abs(y2 - y1))
        # -------------------------------------------------

        # -------------------------------------------------
        GL.glClearColor(1, 1, 1, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        for index, vm_object in self.vm_session.vm_objects_dic.items():
            if vm_object.active:
                #vismol_object has few different types of representations
                '''
                for rep_name in vm_object.representations:
                    # checking all the representations in vismol_object.representations dictionary
                    if vm_object.representations[rep_name] is not None:
                        #  vismol_object.representations[rep_name] may be active or not  True/False
                        if vm_object.representations[rep_name].active:
                            vm_object.representations[rep_name].draw_background_sel_representation()
                '''
                # bachega 03/22/2026
                for rep in vm_object.representations.values():
                    if rep and rep.active:
                        rep.draw_background_sel_representation()
        # -------------------------------------------------

        '''
        # this is an old version
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        #data = GL.glReadPixels(float(pos_x), float(pos_y), width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        # Bachega 03/22/2026
        data = GL.glReadPixels(int(pos_x), int(pos_y), int(width), int(height),
                       GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        data = list(data)
        picked_set = set()
        for i in range(0, len(data), 4):
            #converting RGB values to atoms address (unique id)
            pickedID = data[i] + data[i+1] * 256 + data[i+2] * 256 * 256;
            picked_set.add(pickedID)
        _selected = set()
        for pickedID in picked_set:
            if pickedID == 16777215:
                pass
            else:
                self.atom_picked = self.vm_session.atom_dic_id[pickedID]
                _selected.add(self.vm_session.atom_dic_id[pickedID])
                # The disable variable does not allow, if the selected 
                # atom is already in the selected list, to be removed.
                # The disable variable is "False" for when we use 
                # selection by area (selection box)
                # self.vm_session._selection_function(selected=self.atom_picked, disable=False)
        self.vm_session._selection_function_set(_selected, disable=False)
        self.selection_box_picking = False
        '''


        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)

        # Read pixels (RGBA, unsigned byte)
        data = GL.glReadPixels(
            int(pos_x), int(pos_y),
            int(width), int(height),
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE
        )

        BACKGROUND_ID = 16777215
        atom_dic = self.vm_session.atom_dic_id

        # Decode RGB → ID (ignore alpha)
        picked_ids = {
            data[i] + (data[i+1] << 8) + (data[i+2] << 16)
            for i in range(0, len(data), 4)
        }

        # Map IDs → atoms (filter background + missing IDs)
        selected = {
            atom_dic[pid]
            for pid in picked_ids
            if pid != BACKGROUND_ID and pid in atom_dic
        }

        # Apply selection
        self.vm_session._selection_function_set(selected, disable=False)

        # Disable selection mode
        self.selection_box_picking = False


        
    def _pick(self):
        """
        Perform single-pixel picking using color-encoded IDs.
        """

        BACKGROUND_ID = 16777215

        # Clear buffers before rendering picking scene
        GL.glClearColor(1, 1, 1, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Render selection representations
        for vm_object in self.vm_session.vm_objects_dic.values():
            if not vm_object.active:
                continue

            for rep in vm_object.representations.values():
                if rep and rep.active:
                    rep.draw_background_sel_representation()

        # Ensure tight packing
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)

        # Convert coordinates (GTK → OpenGL)
        x = int(self.picking_x)
        y = int(self.height - self.picking_y)

        # Read single pixel
        data = GL.glReadPixels(x, y, 1, 1, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)

        # Decode RGB → ID
        pickedID = data[0] + data[1] * 256 + data[2] * 256 * 256

        atom_dic = self.vm_session.atom_dic_id

        if pickedID == BACKGROUND_ID:
            self.atom_picked = None

            if self.button == 1:
                self.vm_session._selection_function_set(None)
                self.button = None
        else:
            atom = atom_dic.get(pickedID)

            if atom is not None:
                self.atom_picked = atom

                if self.button == 1:
                    print(atom)
                    self.vm_session._selection_function_set({atom})
                    self.button = None
            else:
                logger.debug(f"pickedID {pickedID} not found")
                self.button = None

        self.picking = False
        return True

    def _pick_old(self):
        """ Function doc """
        GL.glClearColor(1, 1, 1, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        for index, vm_object in self.vm_session.vm_objects_dic.items():
            if vm_object.active:
                #vismol_object has few different types of representations
                for rep_name in vm_object.representations:
                    # checking all the representations in vismol_object.representations dictionary
                    if vm_object.representations[rep_name] is None:
                        pass
                    else:
                        #  vismol_object.representations[rep_name] may be active or not  True/False
                        if vm_object.representations[rep_name].active:
                            vm_object.representations[rep_name].draw_background_sel_representation()
        
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        pos = [self.picking_x, self.height - self.picking_y]
        data = GL.glReadPixels(float(pos[0]), float(pos[1]), 1, 1, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        
        #converting RGB values to atoms address (unique id)
        pickedID = data[0] + data[1] * 256 + data[2] * 256 * 256;
        if pickedID == 16777215:
            self.atom_picked = None
            if self.button == 1:
                self.vm_session._selection_function_set(None)
                self.button = None
        else:
            try:
                # Using antialias, in some rare cases, the pick function is not 
                # identifying the right color of the selected atom. This event is 
                # rare, but can impair viewing if it is not properly ignored
                self.atom_picked = self.vm_session.atom_dic_id[pickedID]
                if self.button == 1:
                    self.vm_session._selection_function_set({self.atom_picked})
                    self.button = None
            except KeyError as ke:
                logger.debug("pickedID {} not found".format(pickedID))
                logger.debug(ke)
                self.button = None
        self.picking = False
        return True
    
    def _get_uniform_location(self, program, name):
        """ Cached wrapper around glGetUniformLocation.

            glGetUniformLocation does a string lookup against the linked
            program every time it is called. Locations don't change once a
            program is linked, so we look each one up once and reuse it for
            the lifetime of the program (cache is cleared in
            create_gl_programs when shaders are recompiled).
        """
        key = (program, name)
        loc = self._uniform_loc_cache.get(key)
        if loc is None:
            loc = GL.glGetUniformLocation(program, name)
            self._uniform_loc_cache[key] = loc
        return loc

    def load_fog(self, program):
        """ Load the fog parameters in the specified program
            
            fog_start -- The coordinates where the fog will begin (always
                         positive)
            fog_end -- The coordinates where the fog will begin (always positive
                       and greater than fog_start)
            fog_color -- The color for the fog (same as background)
        """
        fog_s = self._get_uniform_location(program, "fog_start")
        GL.glUniform1fv(fog_s, 1, self.glcamera.fog_start)
        fog_e = self._get_uniform_location(program, "fog_end")
        GL.glUniform1fv(fog_e, 1, self.glcamera.fog_end)
        fog_c = self._get_uniform_location(program, "fog_color")
        GL.glUniform4fv(fog_c, 1, self.bckgrnd_color)
    
    def _ensure_camera_ubo(self):
        """ Cria o UBO de camera uma unica vez (lazy). Idempotente: chamadas
            seguintes apos a criacao nao fazem nada. Seguro chamar dentro do
            contexto GL (ex.: inicio de render). """
        if self._camera_ubo is not None:
            return
        self._camera_ubo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self._camera_ubo)
        # Aloca o buffer vazio (DYNAMIC: atualizado todo frame)
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, self._camera_ubo_size, None,
                        GL.GL_DYNAMIC_DRAW)
        # Liga o buffer inteiro ao binding point compartilhado
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, self.CAMERA_UBO_BINDING,
                            self._camera_ubo)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

    def update_camera_ubo(self):
        """ Sobe view_mat e proj_mat para o UBO. Chamar UMA vez por frame,
            antes de desenhar as representacoes. Substitui os dois
            glUniformMatrix4fv por-objeto que existiam em load_matrices. """
        self._ensure_camera_ubo()
        view = np.ascontiguousarray(self.glcamera.view_matrix, dtype=np.float32)
        proj = np.ascontiguousarray(self.glcamera.projection_matrix,
                                    dtype=np.float32)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self._camera_ubo)
        # view em offset 0, proj em offset 64 (std140: mat4 ocupa 64 bytes)
        GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, 0,  64, view)
        GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, 64, 64, proj)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

    def _bind_camera_ubo_to_program(self, program):
        """ Liga o bloco 'CameraMatrices' de um programa ao binding point
            compartilhado. Chamar UMA vez por programa, logo apos o link.
            Se o programa nao declara o bloco (ex.: shader ainda nao
            convertido, ou que nao usa camera), glGetUniformBlockIndex
            devolve GL_INVALID_INDEX e simplesmente ignoramos -- isso e o
            que permite a convivencia com shaders nao-convertidos. """
        block_index = GL.glGetUniformBlockIndex(program, "CameraMatrices")
        if block_index != GL.GL_INVALID_INDEX:
            GL.glUniformBlockBinding(program, block_index,
                                     self.CAMERA_UBO_BINDING)

    def load_matrices(self, program=None, model_mat=None):
        """ Load the matrices to OpenGL.
            
            model_mat -- transformation matrix for the objects rendered
            view_mat -- transformation matrix for the camera used
            proj_mat -- matrix for the space to be visualized in the scene

            NOTA (Gargalo 2): para shaders JA CONVERTIDOS ao UBO de camera,
            view_mat e proj_mat vem do UBO (atualizado 1x/frame por
            update_camera_ubo) e NAO sao enviados aqui. Para shaders LEGADOS
            (ainda com 'uniform mat4 view_mat/proj_mat'), mantemos o envio
            individual como antes -- e isso que permite converter um shader
            de cada vez sem quebrar os demais.

            A deteccao usa o cache de uniform location: se 'view_mat' existe
            como uniform individual no programa (loc != -1), e legado.
        """
        model = self._get_uniform_location(program, "model_mat")
        GL.glUniformMatrix4fv(model, 1, GL.GL_FALSE, model_mat)
        # Fallback para shaders nao convertidos: se ainda tem view_mat como
        # uniform individual, envia view+proj como antes.
        view = self._get_uniform_location(program, "view_mat")
        if view != -1:
            GL.glUniformMatrix4fv(view, 1, GL.GL_FALSE,
                                  self.glcamera.view_matrix)
            proj = self._get_uniform_location(program, "proj_mat")
            if proj != -1:
                GL.glUniformMatrix4fv(proj, 1, GL.GL_FALSE,
                                      self.glcamera.projection_matrix)

    def load_dot_params(self, program):
        """ Function doc
        """
        # Extern line
        linewidth = np.float32(80 / abs(self.dist_cam_zrp))
        if linewidth > 3.73:
            linewidth = 3.73
        # Intern line
        antialias = np.float32(80 / abs(self.dist_cam_zrp))
        if antialias > 3.73:
            antialias = 3.73
        # Dot size factor
        dot_factor = np.float32(500 / abs(self.dist_cam_zrp))
        if dot_factor > 150.0:
            dot_factor = 150.0
        uni_vext_linewidth = self._get_uniform_location(program, "vert_ext_linewidth")
        GL.glUniform1fv(uni_vext_linewidth, 1, linewidth)
        uni_vint_antialias = self._get_uniform_location(program, "vert_int_antialias")
        GL.glUniform1fv(uni_vint_antialias, 1, antialias)
        uni_dot_size = self._get_uniform_location(program, "vert_dot_factor")
        GL.glUniform1fv(uni_dot_size, 1, dot_factor)
    
    def load_lights(self, program):
        """ Function doc
        """
        light_pos = self._get_uniform_location(program, "my_light.position")
        GL.glUniform3fv(light_pos, 1, self.light_position)
        amb_coef = self._get_uniform_location(program, "my_light.ambient_coef")
        GL.glUniform1fv(amb_coef, 1, self.light_ambient_coef)
        shiny = self._get_uniform_location(program, "my_light.shininess")
        GL.glUniform1fv(shiny, 1, self.light_shininess)
        intensity = self._get_uniform_location(program, "my_light.intensity")
        GL.glUniform3fv(intensity, 1, self.light_intensity)
    
    def load_antialias_params(self, program):
        """ Function doc """
        a_length = self._get_uniform_location(program, "antialias_length")
        GL.glUniform1fv(a_length, 1, 0.05)
        bck_col = self._get_uniform_location(program, "alias_color")
        GL.glUniform3fv(bck_col, 1, self.bckgrnd_color[:3])
    
    def _draw_labels(self):
        if self.vm_font.vao is None:
            self.vm_font.make_freetype_font()
            #self.vm_font.make_freetype_font(self.vm_session.vm_config.gl_parameters["picking_dots_color"])
            self.vm_font.make_freetype_texture(self.core_shader_programs["freetype"])
        number = 1
        self.chars = 0
        xyz_pos = []
        uv_coords = []
        
        self.do_once = True
        
        if self.do_once:
            for vm_object in self.vm_session.vm_objects_dic.values():
                for index, atom in vm_object.atoms.items():
                    text = atom.residue.name +'/'+ atom.name+'/'+str(atom.index)
                    frame = self._get_vismol_object_frame(atom.vm_object)
                    x, y, z = atom.coords(frame)
                    point = np.array([x, y, z, 1], dtype=np.float32)
                    point = np.dot(point, self.model_mat)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self.vm_font.texture_id)
                    for i, c in enumerate(text):
                        self.chars += 1
                        c_id = ord(c)
                        x = c_id % 16
                        y = c_id // 16 - 2
                        xyz_pos.append(point[0] + i * self.vm_font.char_width)
                        xyz_pos.append(point[1])
                        xyz_pos.append(point[2])
                        uv_coords.append(x * self.vm_font.text_u)
                        uv_coords.append(y * self.vm_font.text_v)
                        uv_coords.append((x + 1) * self.vm_font.text_u)
                        uv_coords.append((y + 1) * self.vm_font.text_v)
                number += 1
            xyz_pos = np.array(xyz_pos, dtype=np.float32)
            uv_coords = np.array(uv_coords, dtype=np.float32)
            
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vm_font.coord_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, xyz_pos.itemsize * len(xyz_pos),
                            xyz_pos, GL.GL_DYNAMIC_DRAW)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vm_font.text_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, uv_coords.itemsize * len(uv_coords),
                            uv_coords, GL.GL_DYNAMIC_DRAW)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glUseProgram(self.core_shader_programs["freetype"])
            self.do_once = False
        
        self.vm_font.load_matrices(self.core_shader_programs["freetype"],
                                   self.glcamera.view_matrix,
                                   self.glcamera.projection_matrix)
        self.vm_font.load_font_params(self.core_shader_programs["freetype"])
        
        GL.glBindVertexArray(self.vm_font.vao)
        GL.glDrawArrays(GL.GL_POINTS, 0, self.chars)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def _draw_distance_labels(self, vm_object):
        if self.vm_font_dist.vao is None:
            self.vm_font_dist.color = np.array(self.vm_session.vm_config.gl_parameters["pk_dist_label_color"], dtype=np.float32)
            self.vm_font_dist.make_freetype_font()
            self.vm_font_dist.make_freetype_texture(self.core_shader_programs["freetype"])
        
        number = 1.5
        chars = 0
        xyz_pos = []
        uv_coords = []
        self.vm_font_dist.char_res    =15
        #self.vm_font_dist.char_width  = 0.15
        #self.vm_font_dist.char_height = 0.15
        text =  '{:.2f}'.format(vm_object.dist)
        #frame = self._get_vismol_object_frame(atom.vm_object)
        #x, y, z = atom.coords(frame)
        x, y, z = vm_object.midpoint[0],vm_object.midpoint[1],vm_object.midpoint[2]

        point = np.array([x, y, z, 1], dtype=np.float32)
        point = np.dot(point, self.model_mat)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.vm_font_dist.texture_id)
        for i, c in enumerate(text):
            chars += 1
            c_id = ord(c)
            x = c_id % 16
            y = c_id // 16 - 2
            xyz_pos.append(point[0] + i * self.vm_font_dist.char_width -0.2) # -0.2 is shift factor to drag the label to center  
            xyz_pos.append(point[1])
            xyz_pos.append(point[2])
            uv_coords.append(x * self.vm_font_dist.text_u)
            uv_coords.append(y * self.vm_font_dist.text_v)
            uv_coords.append((x + 1) * self.vm_font_dist.text_u)
            uv_coords.append((y + 1) * self.vm_font_dist.text_v)
        
        xyz_pos = np.array(xyz_pos, dtype=np.float32)
        uv_coords = np.array(uv_coords, dtype=np.float32)
        
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vm_font_dist.coord_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, xyz_pos.itemsize * len(xyz_pos),
                        xyz_pos, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vm_font_dist.text_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, uv_coords.itemsize * len(uv_coords),
                        uv_coords, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glUseProgram(self.core_shader_programs["freetype"])
        
        self.vm_font_dist.load_matrices(self.core_shader_programs["freetype"],
                                   self.glcamera.view_matrix,
                                   self.glcamera.projection_matrix)
        self.vm_font_dist.load_font_params(self.core_shader_programs["freetype"])
        
        GL.glBindVertexArray(self.vm_font_dist.vao)
        GL.glDrawArrays(GL.GL_POINTS, 0, chars)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        
        '''
        if self.vm_font_dist_static.vao is None:
            self.vm_font_dist_static.make_freetype_font()
            self.vm_font_dist_static.make_freetype_texture(self.core_shader_programs["freetype"])
        
        number     = 1
        self.chars = 0
        point      = vm_object.midpoint
        xyz_pos    = []
        uv_coords  = []
        
        
        
        self.do_once = True
        
        #if self.do_once:

        text = "distance: {}".format(0.0) 
        point = np.array([point[0],point[1],point[2], 1 ], dtype=np.float32)
        #point = np.dot(point, self.model_mat)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.vm_font_dist.texture_id)
        for i, c in enumerate(text):
            self.chars += 1
            c_id = ord(c)
            x = c_id % 16
            y = c_id // 16 - 2
            xyz_pos.append(point[0] + i * self.vm_font_dist.char_width)
            xyz_pos.append(point[1])
            xyz_pos.append(point[2])
            uv_coords.append(x * self.vm_font_dist.text_u)
            uv_coords.append(y * self.vm_font_dist.text_v)
            uv_coords.append((x + 1) * self.vm_font_dist.text_u)
            uv_coords.append((y + 1) * self.vm_font_dist.text_v)
            number += 1
        xyz_pos = np.array(xyz_pos, dtype=np.float32)
        uv_coords = np.array(uv_coords, dtype=np.float32)
        
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vm_font_dist.coord_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, xyz_pos.itemsize * len(xyz_pos),
                        xyz_pos, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vm_font_dist.text_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, uv_coords.itemsize * len(uv_coords),
                        uv_coords, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glUseProgram(self.core_shader_programs["freetype"])
        
        self.vm_font_dist.load_matrices(self.core_shader_programs["freetype"],
                                   self.glcamera.view_matrix,
                                   self.glcamera.projection_matrix)
        self.vm_font_dist.load_font_params(self.core_shader_programs["freetype"])
        
        GL.glBindVertexArray(self.vm_font_dist.vao)
        GL.glDrawArrays(GL.GL_POINTS, 0, self.chars)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        #print('aqui')
        '''
    
    def _draw_picking_label(self):
        """ This function draws the labels of the atoms selected by the
            function picking #1 #2 #3 #4
        """
        if self.vm_font.vao is None:
            #self.vm_font.make_freetype_font(color = self.vm_session.vm_config.gl_parameters["picking_dots_color"])
            self.vm_font.color = np.array(self.vm_session.vm_config.gl_parameters["pk_label_color"], dtype=np.float32)
            self.vm_font.make_freetype_font()
            self.vm_font.make_freetype_texture(self.core_shader_programs["freetype"])
        
        number = 1
        chars = 0
        xyz_pos = []
        uv_coords = []
        
        for atom in self.vm_session.picking_selections.picking_selections_list:
            if atom:
                text = "#" + str(number)
                frame = self._get_vismol_object_frame(atom.vm_object)
                x, y, z = atom.coords(frame)
                

                point = np.array([x, y, z, 1], dtype=np.float32)
                point = np.dot(point, self.model_mat)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.vm_font.texture_id)
                for i, c in enumerate(text):
                    chars += 1
                    c_id = ord(c)
                    x = c_id % 16
                    y = c_id // 16 - 2
                    xyz_pos.append(point[0] + i * self.vm_font.char_width -0.075)
                    xyz_pos.append(point[1]-0.05)
                    xyz_pos.append(point[2])
                    uv_coords.append(x * self.vm_font.text_u)
                    uv_coords.append(y * self.vm_font.text_v)
                    uv_coords.append((x + 1) * self.vm_font.text_u)
                    uv_coords.append((y + 1) * self.vm_font.text_v)
            number += 1
        xyz_pos = np.array(xyz_pos, dtype=np.float32)
        uv_coords = np.array(uv_coords, dtype=np.float32)
        
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vm_font.coord_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, xyz_pos.itemsize * len(xyz_pos),
                        xyz_pos, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vm_font.text_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, uv_coords.itemsize * len(uv_coords),
                        uv_coords, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glUseProgram(self.core_shader_programs["freetype"])
        
        self.vm_font.load_matrices(self.core_shader_programs["freetype"],
                                   self.glcamera.view_matrix,
                                   self.glcamera.projection_matrix)
        self.vm_font.load_font_params(self.core_shader_programs["freetype"])
        
        GL.glBindVertexArray(self.vm_font.vao)
        GL.glDrawArrays(GL.GL_POINTS, 0, chars)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        
        '''
        atomlist = self.vm_session.picking_selections.picking_selections_list
        
        text_d1 = ""
        text_d2 = ""
        text_d3 = ""
        angle_1 = ""
        # this should be a new function later
        if atomlist[0] and atomlist[1]:
            
            crd1 = atomlist[0].coords(frame)
            crd2 = atomlist[1].coords(frame)
            d1 = ((crd2[0]-crd1[0])**2+ 
                 (crd2[1]-crd1[1])**2+ 
                 (crd2[2]-crd1[2])**2)**0.5
            
            text_d1 = "dist 1-2: {:7.5f} ".format(d1)
            
            
            if atomlist[2]:
                #print('distance #1 - #2: ', d)
                crd3 = atomlist[2].coords(frame)
                
                d2 = ((crd2[0]-crd3[0])**2+ 
                     ( crd2[1]-crd3[1])**2+ 
                     ( crd2[2]-crd3[2])**2)**0.5
                
                text_d2 = "dist 2-3: {:7.5f} ".format(d2)
                
                
                v1 = [crd2[0]-crd1[0], 
                      crd2[1]-crd1[1], 
                      crd2[2]-crd1[2]]
                v2 = [crd2[0]-crd3[0], 
                      crd2[1]-crd3[1], 
                      crd2[2]-crd3[2]]
                dot_product = sum(v1[i] * v2[i] for i in range(len(v1)))  
                
                magnitude_a = math.sqrt(sum(x**2 for x in v1))
                magnitude_b = math.sqrt(sum(x**2 for x in v2))
                cosine_theta = dot_product / (magnitude_a * magnitude_b)
                angle_rad = math.acos(cosine_theta)
                angle_deg = math.degrees(angle_rad)
                
                angle_1 = "angle 1-2-3: {:7.5f} ".format(angle_deg)
                
        print(text_d1, text_d2, text_d3, angle_1)
                #print(angle_deg)
        '''

        '''
        self._draw_distance_labels( distance = d)
        '''

    def _compile_shader_picking_dots(self):
        """ Function doc """
        
        safe =  self.vm_config.gl_parameters["picking_dots_safe"]
        if safe:
            self.core_shader_programs["picking_dots"] = self.load_shaders(shaders_pick.vertex_shader_picking_dots_safe,
                                                                          shaders_pick.fragment_shader_picking_dots_safe)
        else:
            self.core_shader_programs["picking_dots"] = self.load_shaders(shaders_pick.vertex_shader_picking_dots,
                                                                      shaders_pick.fragment_shader_picking_dots)
        
    def _compile_shader_dots(self):
        """ Function doc """
        dot_type = self.vm_config.gl_parameters["dot_type"]
        self.shader_programs["dots"] = self.load_shaders(shaders_dots.shader_type[dot_type]["vertex_shader"],
                                                shaders_dots.shader_type[dot_type]["fragment_shader"])
        self.shader_programs["dots_sel"] = self.load_shaders(shaders_dots.shader_type[dot_type]["sel_vertex_shader"],
                                                    shaders_dots.shader_type[dot_type]["sel_fragment_shader"])
    
    def _compile_shader_lines(self):
        """ Function doc """
        line_type = self.vm_config.gl_parameters["line_type"]
        self.shader_programs["lines"] = self.load_shaders(shaders_lines.shader_type[line_type]["vertex_shader"],
                                                  shaders_lines.shader_type[line_type]["fragment_shader"],
                                                  shaders_lines.shader_type[line_type]["geometry_shader"])
        self.shader_programs["lines_sel"] = self.load_shaders(shaders_lines.shader_type[line_type]["sel_vertex_shader"],
                                                      shaders_lines.shader_type[line_type]["sel_fragment_shader"],
                                                      shaders_lines.shader_type[line_type]["sel_geometry_shader"])
        '''
        #sticks_type = self.vm_config.gl_parameters["sticks_type"]
        #self.shader_programs["lines"] = self.load_shaders(shaders_sticks.shader_type[sticks_type]["vertex_shader"],
        #                                           shaders_sticks.shader_type[sticks_type]["fragment_shader"],
        #                                           shaders_sticks.shader_type[sticks_type]["geometry_shader"])
        #self.shader_programs["lines_sel"] = self.load_shaders(shaders_sticks.shader_type[sticks_type]["sel_vertex_shader"],
        #                                               shaders_sticks.shader_type[sticks_type]["sel_fragment_shader"],
        #                                               shaders_sticks.shader_type[sticks_type]["sel_geometry_shader"])
        '''
        
    def _compile_shader_nonbonded(self):
        """ Function doc """
        self.shader_programs["nonbonded"] = self.load_shaders(shaders_nonbonded.vertex_shader_non_bonded,
                                                      shaders_nonbonded.fragment_shader_non_bonded,
                                                      shaders_nonbonded.geometry_shader_non_bonded)
        self.shader_programs["nonbonded_sel"] = self.load_shaders(shaders_nonbonded.sel_vertex_shader_non_bonded,
                                                          shaders_nonbonded.sel_fragment_shader_non_bonded,
                                                          shaders_nonbonded.sel_geometry_shader_non_bonded)
    
    def _compile_shader_sticks(self):
        """ Function doc """
        sticks_type = self.vm_config.gl_parameters["sticks_type"]
        self.shader_programs["sticks"] = self.load_shaders(shaders_sticks.shader_type[sticks_type]["vertex_shader"],
                                                   shaders_sticks.shader_type[sticks_type]["fragment_shader"],
                                                   shaders_sticks.shader_type[sticks_type]["geometry_shader"])
        self.shader_programs["sticks_sel"] = self.load_shaders(shaders_sticks.shader_type[sticks_type]["sel_vertex_shader"],
                                                       shaders_sticks.shader_type[sticks_type]["sel_fragment_shader"],
                                                       shaders_sticks.shader_type[sticks_type]["sel_geometry_shader"])
    
    def _compile_shader_ribbons(self):
        """ Function doc """
        #sticks_type = self.vm_config.gl_parameters["ribbon_type"]
        sticks_type = self.vm_config.gl_parameters["sticks_type"]
        self.shader_programs["ribbons"] = self.load_shaders(shaders_sticks.shader_type[sticks_type]["vertex_shader"],
                                                   shaders_sticks.shader_type[sticks_type]["fragment_shader"],
                                                   shaders_sticks.shader_type[sticks_type]["geometry_shader"])
        self.shader_programs["ribbons_sel"] = self.load_shaders(shaders_sticks.shader_type[sticks_type]["sel_vertex_shader"],
                                                       shaders_sticks.shader_type[sticks_type]["sel_fragment_shader"],
                                                       shaders_sticks.shader_type[sticks_type]["sel_geometry_shader"])
    
    def _compile_shader_dynamic(self):
        """ Function doc """
        sticks_type = self.vm_config.gl_parameters["sticks_type"]
        self.shader_programs["dynamic"] = self.load_shaders(shaders_sticks.shader_type[sticks_type]["vertex_shader"],
                                                   shaders_sticks.shader_type[sticks_type]["fragment_shader"],
                                                   shaders_sticks.shader_type[sticks_type]["geometry_shader"])
        self.shader_programs["dynamic_sel"] = self.load_shaders(shaders_sticks.shader_type[sticks_type]["sel_vertex_shader"],
                                                       shaders_sticks.shader_type[sticks_type]["sel_fragment_shader"],
                                                       shaders_sticks.shader_type[sticks_type]["sel_geometry_shader"])
    
    def _compile_shader_spheres(self):
        """ Function doc """
        self.shader_programs["spheres"] = self.load_shaders(shaders_spheres.vertex_shader_spheres,
                                                    shaders_spheres.fragment_shader_spheres)
        self.shader_programs["spheres_sel"] = self.load_shaders(shaders_spheres.sel_vertex_shader_spheres,
                                                        shaders_spheres.sel_fragment_shader_spheres)
    
    def _compile_shader_picking_spheres(self):
        """ Function doc """
        #print('\npicking_spheres'*10)
        #self.shader_programs["picking_spheres"] = self.load_shaders(shaders_spheres.vertex_shader_spheres,
        #                                                            shaders_spheres.fragment_shader_spheres)
        #self.shader_programs["picking_spheres_sel"] = self.load_shaders(shaders_spheres.vertex_shader_picking_spheres,
        #                                                                shaders_spheres.fragment_shader_picking_spheres)
        #
        self.shader_programs["picking_spheres"] = self.load_shaders(shaders_spheres.vertex_shader_picking_spheres,
                                                                    shaders_spheres.fragment_shader_picking_spheres)
        self.shader_programs["picking_spheres_sel"] = self.load_shaders(shaders_spheres.vertex_shader_picking_spheres,
                                                                        shaders_spheres.fragment_shader_picking_spheres)
    
    def _compile_shader_vdw_spheres(self):
        """ Function doc """
        self.shader_programs["vdw_spheres"] = self.load_shaders(shaders_spheres.sel_vertex_shader_spheres,
                                                    shaders_spheres.sel_fragment_shader_spheres)
        self.shader_programs["vdw_spheres_sel"] = self.load_shaders(shaders_spheres.sel_vertex_shader_spheres,
                                                        shaders_spheres.sel_fragment_shader_spheres)
    
    def _compile_shader_dash(self):
        """ Function doc """
        self.shader_programs["dash"] = self.load_shaders(shaders_dashed_lines.vertex_shader_dashed_lines,
                                                         shaders_dashed_lines.fragment_shader_dashed_lines,
                                                         shaders_dashed_lines.geometry_shader_dashed_lines)
        self.shader_programs["dash_sel"] = self.load_shaders( shaders_dashed_lines.sel_vertex_shader_dashed_lines,
                                                              shaders_dashed_lines.sel_fragment_shader_dashed_lines,
                                                              shaders_dashed_lines.sel_geometry_shader_dashed_lines)
    
    def _compile_shader_freetype(self):
        """ Function doc """
        self.core_shader_programs["freetype"] = self.load_shaders(shaders_vm_freetype.vertex_shader_freetype,
                                                         shaders_vm_freetype.fragment_shader_freetype,
                                                         shaders_vm_freetype.geometry_shader_freetype)
    
    def _compile_shader_static_freetype(self):
        """ Function doc """
        self.core_shader_programs["static_freetype"] = self.load_shaders(shaders_vm_freetype.static_vertex_shader_freetype,
                                                                         shaders_vm_freetype.static_fragment_shader_freetype,
                                                                         shaders_vm_freetype.static_geometry_shader_freetype)

    def _compile_shader_impostor(self):
        """ Function doc """
        im_type = self.vm_config.gl_parameters["impostor_type"]
        self.shader_programs["impostor"] = self.load_shaders(shaders_impostor.shader_type[im_type]["vertex_shader"],
                                                     shaders_impostor.shader_type[im_type]["fragment_shader"],
                                                     shaders_impostor.shader_type[im_type]["geometry_shader"])
        self.shader_programs["impostor_sel"] = self.load_shaders(shaders_impostor.shader_type[im_type]["sel_vertex_shader"],
                                                        shaders_impostor.shader_type[im_type]["sel_fragment_shader"],
                                                        shaders_impostor.shader_type[im_type]["sel_geometry_shader"])
    
    def _compile_shader_surface(self):
        """ Function doc """
        #self.shader_programs["surface"] = self.load_shaders(shaders_surface.vertex_shader_surface,
        #                                            shaders_surface.fragment_shader_surface,
        #                                            shaders_surface.geometry_shader_surface)
        
        '''
        self.shader_programs["surface"] = self.load_shaders(shaders_surface.vertex_shader_lines,
                                                    shaders_surface.fragment_shader_lines,
                                                    shaders_surface.geometry_shader_lines)
        self.shader_programs["surface_sel"] = self.load_shaders(shaders_spheres.vertex_shader_spheres,
                                                        shaders_spheres.fragment_shader_spheres)
        #'''
        
        self.shader_programs["surface"] = self.load_shaders(shaders_surface.vertex_shader_lines,
                                                    shaders_surface.fragment_shader_lines,
                                                    shaders_surface.geometry_shader_lines)
        self.shader_programs["surface_sel"] = self.load_shaders(shaders_spheres.vertex_shader_spheres,
                                                        shaders_spheres.fragment_shader_spheres)
    
    
    
    
    '''
    def _compile_shader_cartoon(self):
        """ Function doc """
        self.shader_programs["cartoon"] = self.load_shaders(shaders_cartoon.v_shader_triangles,
                                                    shaders_cartoon.f_shader_triangles)
        self.shader_programs["cartoon_sel"] = self.load_shaders(shaders_cartoon.v_shader_triangles,
                                                    shaders_cartoon.f_shader_triangles)
    
    
    #'''
    
    '''
    #----------------------------NOT IMPLEMENTED YET---------------------------#
    # def _dynamic_bonds_shaders(self):
    #     """ Function doc """
    #     self.shader_programs["dynamic"] = self.load_shaders(shaders_sticks.vertex_shader_sticks,
    #                                                 shaders_sticks.fragment_shader_sticks,
    #                                                 shaders_sticks.geometry_shader_sticks)
    #     self.shader_programs["dynamic_sel"] = self.load_shaders(shaders_sticks.sel_vertex_shader_sticks,
    #                                                    shaders_sticks.sel_fragment_shader_sticks,
    #                                                    shaders_sticks.sel_geometry_shader_sticks)
    
    # def _wires_dot_shaders(self):
    #     """ Function doc """
    #     self.shader_programs["wires"] = self.load_shaders(shaders_wires.vertex_shader_wires,
    #                                               shaders_wires.fragment_shader_wires,
    #                                               shaders_wires.geometry_shader_wires)
    #     self.shader_programs["wires_sel"] = self.load_shaders(shaders_spheres.vertex_shader_spheres,
    #                                                     shaders_spheres.fragment_shader_spheres)
    #----------------------------NOT IMPLEMENTED YET---------------------------#
    #'''
    
    def _safe_frame_coords(self, vismol_object):
        """ Function doc 
        This function checks if the number of the called frame will not exceed 
        the limit of frames that each object has. Allowing two objects with 
        different trajectory sizes to be manipulated at the same time within the 
        glArea
        """
        if self.vm_session.frame < 0:
            self.vm_session.frame = 0
        if self.vm_session.frame >= vismol_object.frames.shape[0] - 1:
            frame_coords = vismol_object.frames[vismol_object.frames.shape[0] - 1]
            frame = vismol_object.frames.shape[0] - 1
        else:
            frame_coords = vismol_object.frames[self.vm_session.frame]
            frame = self.vm_session.frame
        return frame_coords, frame
    
    def _get_vismol_object_frame(self, vismol_object):
        """ Function doc """
        if self.vm_session.frame < 0:
            self.vm_session.frame = 0
        if self.vm_session.frame >= vismol_object.frames.shape[0] - 1:
            frame = vismol_object.frames.shape[0] - 1
        else:
            frame = self.vm_session.frame
        return frame
    
    def get_viewport_pos(self, x, y):
        """ Function doc """
        px = (2.0 * x - self.width) / self.width
        py = (2.0 * y - self.height) / self.height
        return np.array([px, -py], dtype=np.float32)
    
    def _mouse_pos_old(self, x, y):
        """
        Use the ortho projection and viewport information
        to map from mouse co-ordinates back into world
        co-ordinates
        """
        px = x / self.width
        py = y / self.height
        px = self.left + px * (self.right - self.left)
        py = self.top + py * (self.bottom - self.top)
        pz = self.glcamera.z_near
        return px, py, pz

    def _mouse_pos(self, x, y):
        """
        Convert screen (mouse) coordinates to world coordinates
        using orthographic projection.

        Assumes:
        - Screen origin is top-left (e.g., GTK)
        - OpenGL origin is bottom-left
        """

        # Avoid division by zero
        if self.width == 0 or self.height == 0:
            return None

        # Normalize to [0, 1]
        nx = float(x) / float(self.width)
        ny = 1.0 - (float(y) / float(self.height))  # invert Y axis

        # Map to world coordinates
        world_x = self.left + nx * (self.right - self.left)
        world_y = self.bottom + ny * (self.top - self.bottom)

        # Use near plane as default Z
        world_z = self.glcamera.z_near

        return world_x, world_y, world_z

    def center_on_atom(self, atom):
        """ Function doc
        """
        frame_index = self._get_vismol_object_frame(atom.vm_object)
        self.center_on_coordinates(atom.vm_object, atom.coords(frame_index))

    def center_on_coordinates_new(self, vismol_object, target):
        """
        Smoothly translate all objects so that the target coordinate becomes centered.
        
        Takes the coordinates of an atom in absolute coordinates and first
        transforms them in 4D world coordinates, then takes the unit vector
        of that atom position to generate the loop animation. To generate
        the animation, first obtains the distance from the zero reference
        point (always 0,0,0) to the atom, then divides this distance in a
        defined number of cycles, this result will be the step for
        translation. For the translation, the world will move a number of
        steps defined, and every new point will be finded by multiplying the
        unit vector by the step. As a final step, to avoid biases, the world
        will be translated to the atom position in world coordinates.
        The effects will be applied on the model matrices of every VisMol
        object and the model matrix of the window.
        
        """

        if not np.allclose(self.zero_reference_point, target):
            self.zero_reference_point[:] = target

            target_pos_4d = np.array([*target, 1.0], dtype=np.float32)

            # Transform to world space
            target_in_world = vismol_object.model_mat.T.dot(target_pos_4d)[:3]

            norm = np.linalg.norm(target_in_world)
            if norm == 0:
                return

            unit_vec = target_in_world / norm

            steps = self.vm_config.gl_parameters.get("center_steps", 15)
            step_size = norm / steps

            def iter_all_objects():
                yield from self.vm_session.vm_objects_dic.values()
                for obj in self.vm_session.vm_geometric_object_dic.values():
                    if obj:
                        yield obj

            # Animate movement
            for _ in range(steps):
                delta = unit_vec * step_size

                for obj in iter_all_objects():
                    obj.model_mat = mop.my_glTranslatef(obj.model_mat, -delta)

                # Trigger redraw (GTK)
                if self.vm_session.toolkit == "Gtk_3.0":
                    win = self.parent_widget.get_window()
                    win.invalidate_rect(None, False)
                    win.process_updates(False)
                else:
                    raise RuntimeError("Not implemented for Qt5 yet")

                time.sleep(self.vm_config.gl_parameters["center_on_coord_sleep_time"])

            # Final correction to eliminate accumulated error
            for obj in iter_all_objects():
                final_pos = obj.model_mat.T.dot(target_pos_4d)[:3]
                obj.model_mat = mop.my_glTranslatef(obj.model_mat, -final_pos)

            self.parent_widget.queue_draw()

    def center_on_coordinates(self, vismol_object, target):
        """ Takes the coordinates of an atom in absolute coordinates and first
            transforms them in 4D world coordinates, then takes the unit vector
            of that atom position to generate the loop animation. To generate
            the animation, first obtains the distance from the zero reference
            point (always 0,0,0) to the atom, then divides this distance in a
            defined number of cycles, this result will be the step for
            translation. For the translation, the world will move a number of
            steps defined, and every new point will be finded by multiplying the
            unit vector by the step. As a final step, to avoid biases, the world
            will be translated to the atom position in world coordinates.
            The effects will be applied on the model matrices of every VisMol
            object and the model matrix of the window.
        """
        if (self.zero_reference_point[0] != target[0]) or \
           (self.zero_reference_point[1] != target[1]) or \
           (self.zero_reference_point[2] != target[2]):
            self.zero_reference_point[:] = target
            pos = np.array([target[0],target[1],target[2],1], dtype=np.float32)
            model_pos = vismol_object.model_mat.T.dot(pos)[:3]
            self.model_mat = mop.my_glTranslatef(self.model_mat, -model_pos)
            unit_vec = model_pos / np.linalg.norm(model_pos)
            step = np.linalg.norm(model_pos)/15.0
            for i in range(15):
                to_move = unit_vec * step
                
                for index, vm_object in self.vm_session.vm_objects_dic.items():
                    vm_object.model_mat = mop.my_glTranslatef(vm_object.model_mat, -to_move)
                
                for key in self.vm_session.vm_geometric_object_dic.keys():
                    vm_object = self.vm_session.vm_geometric_object_dic[key]
                    if vm_object:
                        vm_object.model_mat = mop.my_glTranslatef(vm_object.model_mat, -to_move)
                
                # WARNING: Method only works with GTK!!!
                if self.vm_session.toolkit == "Gtk_3.0":
                    self.parent_widget.get_window().invalidate_rect(None, False)
                    self.parent_widget.get_window().process_updates(False)
                elif self.vm_session.toolkit == "Qt5":
                    logger.critical("Not implemented for Qt5 yet :(")
                    raise RuntimeError("Not implemented for Qt5 yet :(")
                # WARNING: Method only works with GTK!!!
                time.sleep(self.vm_config.gl_parameters["center_on_coord_sleep_time"])
            
            for index, vm_object in self.vm_session.vm_objects_dic.items():
                model_pos = vm_object.model_mat.T.dot(pos)[:3]
                vm_object.model_mat = mop.my_glTranslatef(vm_object.model_mat, -model_pos)
            
            for key in self.vm_session.vm_geometric_object_dic.keys():
                vm_object = self.vm_session.vm_geometric_object_dic[key]
                if vm_object:
                    model_pos = vm_object.model_mat.T.dot(pos)[:3]
                    vm_object.model_mat = mop.my_glTranslatef(vm_object.model_mat, -model_pos)
            #self.dragging = True
            self.parent_widget.queue_draw()
    
    def queue_draw(self):
        """ Function doc """
        self.parent_widget.queue_draw()
