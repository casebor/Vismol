#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  representations.py
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

import ctypes
import numpy as np
from OpenGL import GL
from logging import getLogger
from vismol.libgl.vismol_font import VismolFont

logger = getLogger(__name__)


class Representation:
    """ Class doc """
    
    def __init__ (self, vismol_object, vismol_glcore, name, active, indexes, is_dynamic = False):
        self.vm_object = vismol_object
        self.vm_session = vismol_object.vm_session
        self.vm_glcore = vismol_glcore
        self.name = name
        self.active = active
        self.indexes = np.array(indexes, dtype=np.uint32)
        self.elements = np.uint32(self.indexes.shape[0])
        
        self.is_dynamic = is_dynamic
        
        self.was_rep_modified = False
        self.was_sel_modified = False
        self.was_col_modified = False
        self.was_rep_coord_modified = False
        self.was_sel_coord_modified = False
        self.was_rep_ind_modified = False
        self.was_sel_ind_modified = False
        self.was_rep_col_modified = False
        # representation
        self.vao = None
        self.ind_vbo = None
        self.coord_vbo = None
        self.col_vbo = None
        self.size_vbo = None
        # selection
        self.sel_vao = None
        self.sel_ind_vbo = None
        self.sel_coord_vbo = None
        self.sel_col_vbo = None
        self.sel_size_vbo = None
        # shaders
        self.shader_program = None
        self.sel_shader_program = None
    
    def _check_vao_and_vbos(self):
        #print(self.name)
        self.shader_program = self.vm_glcore.shader_programs[self.name]
        self.sel_shader_program = self.vm_glcore.shader_programs[self.name + "_sel"]
        if self.vao is None:
            self._make_gl_representation_vao_and_vbos()
        if self.sel_vao is None:
            self._make_gl_sel_representation_vao_and_vbos()
    
    def _make_gl_representation_vao_and_vbos(self):
        """ Function doc """
        logger.debug("building '{}' representation VAO and VBOs".format(self.name))
        self.vao = self._make_gl_vao()
        self.ind_vbo = self._make_gl_index_buffer(self.indexes)
        self.coord_vbo = self._make_gl_coord_buffer(self.vm_object.frames[0], self.shader_program)
        self.col_vbo = self._make_gl_color_buffer(self.vm_object.colors, self.shader_program)
    
    def _make_gl_sel_representation_vao_and_vbos(self):
        """ Function doc """
        logger.debug("building '{}' background selection VAO and VBOs".format(self.name))
        self.sel_vao = self._make_gl_vao()
        self.sel_ind_vbo = self._make_gl_index_buffer(self.indexes)
        self.sel_coord_vbo = self._make_gl_coord_buffer(self.vm_object.frames[0], self.sel_shader_program)
        self.sel_col_vbo = self._make_gl_color_buffer(self.vm_object.color_indexes, self.sel_shader_program)
    
    def _make_gl_vao(self):
        """ Function doc """
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        return vao
    
    def _make_gl_index_buffer(self, indexes):
        """ Function doc """
        ind_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ind_vbo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indexes.nbytes, indexes, GL.GL_DYNAMIC_DRAW)
        return ind_vbo
    
    def _make_gl_coord_buffer(self, coords, program):
        """ Function doc """
        coord_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, coord_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, coords.nbytes, coords, GL.GL_STATIC_DRAW)
        att_position = GL.glGetAttribLocation(program, "vert_coord")
        GL.glEnableVertexAttribArray(att_position)
        GL.glVertexAttribPointer(att_position, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*coords.itemsize, ctypes.c_void_p(0))
        return coord_vbo
    
    def _make_gl_color_buffer(self, colors, program, instances=False):
        """ Function doc """
        col_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, col_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.nbytes, colors, GL.GL_STATIC_DRAW)
        att_colors = GL.glGetAttribLocation(program, "vert_color")
        GL.glEnableVertexAttribArray(att_colors)
        GL.glVertexAttribPointer(att_colors, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*colors.itemsize, ctypes.c_void_p(0))
        if instances:
            GL.glVertexAttribDivisor(att_colors, 1)
        return col_vbo
    
    def _make_gl_radius_buffer(self, radii, program, instances=False):
        """ Function doc """
        rad_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, rad_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, radii.nbytes, radii, GL.GL_STATIC_DRAW)
        att_rads = GL.glGetAttribLocation(program, "vert_radius")
        GL.glEnableVertexAttribArray(att_rads)
        GL.glVertexAttribPointer(att_rads, 1, GL.GL_FLOAT, GL.GL_FALSE, radii.itemsize, ctypes.c_void_p(0))
        if instances:
            GL.glVertexAttribDivisor(att_rads, 1)
        return rad_vbo
    
    def _make_gl_instance_buffer(self, instances, program):
        """ Function doc """
        insta_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, insta_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, instances.nbytes, instances, GL.GL_STATIC_DRAW)
        gl_insta = GL.glGetAttribLocation(program, "vert_instance")
        GL.glEnableVertexAttribArray(gl_insta)
        GL.glVertexAttribPointer(gl_insta, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        GL.glVertexAttribDivisor(gl_insta, 1)
        return insta_vbo
    
    def _make_gl_impostor_buffer(self, impostors_radii, program):
        """ Function doc """
        size_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, size_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, impostors_radii.nbytes, impostors_radii, GL.GL_STATIC_DRAW)
        att_size = GL.glGetAttribLocation(program, "vert_dot_size")
        GL.glEnableVertexAttribArray(att_size)
        GL.glVertexAttribPointer(att_size, 1, GL.GL_FLOAT, GL.GL_FALSE, impostors_radii.itemsize, ctypes.c_void_p(0))
        
        self.ratio = self.vm_glcore.width / self.vm_glcore.height
        ratio_vbo = 1
        # ratio = np.repeat(self.ratio, impostors_radii.shape[0])
        # ratio_vbo = GL.glGenBuffers(1)
        # GL.glBindBuffer(GL.GL_ARRAY_BUFFER, ratio_vbo)
        # GL.glBufferData(GL.GL_ARRAY_BUFFER, ratio.nbytes, ratio, GL.GL_STATIC_DRAW)
        # att_ratio = GL.glGetAttribLocation(program, "hw_ratio")
        # GL.glEnableVertexAttribArray(att_ratio)
        # GL.glVertexAttribPointer(att_ratio, 1, GL.GL_FLOAT, GL.GL_FALSE, ratio.itemsize, ctypes.c_void_p(0))
        return size_vbo, ratio_vbo
    
    def _load_coord_vbo(self, coord_vbo=False, sel_coord_vbo=False):
        """ This function assigns the coordinates to 
        be drawn by the function  draw_representation"""
        frame, f = self.vm_glcore._safe_frame_coords(self.vm_object)
        if coord_vbo:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.coord_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, frame.nbytes, frame, GL.GL_STATIC_DRAW)
        
        if sel_coord_vbo:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.sel_coord_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, frame.nbytes, frame, GL.GL_STATIC_DRAW)
    
    def _load_ind_vbo(self, ind_vbo=False, sel_ind_vbo=False):
        """ Function doc """
        if self.is_dynamic:
            frame, f = self.vm_glcore._safe_frame_coords(self.vm_object)
            self.define_new_indexes_to_vbo(input_indexes = self.vm_object.dynamic_bonds[f])
            
        if ind_vbo:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ind_vbo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.indexes.nbytes, self.indexes, GL.GL_DYNAMIC_DRAW)
        
        if sel_ind_vbo:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.sel_ind_vbo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.indexes.nbytes, self.indexes, GL.GL_DYNAMIC_DRAW)
    
    def _load_color_vbo(self, colors):
        """ This function assigns the colors to
            be drawn by the function  draw_representation"""
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.col_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.nbytes, colors, GL.GL_STATIC_DRAW)
    
    def _enable_anti_alias_to_lines(self):
        """ Function doc """
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
    
    def _disable_anti_alias_to_lines(self):
        """ Function doc """
        GL.glDisable(GL.GL_LINE_SMOOTH)
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_DEPTH_TEST)
    
    def define_new_indexes_to_vbo(self, input_indexes):
        """ Function doc """
        self.indexes = np.array(input_indexes, dtype=np.uint32)
        self.elements = np.uint32(self.indexes.shape[0])


class PickingDotsRepresentation(Representation):
    """ Class doc """
    
    def __init__(self, vismol_object, vismol_glcore, indexes=None, active=True):
        """ Class initialiser """
        super(PickingDotsRepresentation, self).__init__(vismol_object, vismol_glcore, "picking_dots", active, indexes)
    
    def _check_vao_and_vbos(self):
        self.shader_program = self.vm_glcore.core_shader_programs[self.name]
        if self.vao is None:
            self._make_gl_representation_vao_and_vbos()
    
    def _make_gl_representation_vao_and_vbos(self):
        """ Function doc """
        logger.debug("building '{}' representation VAO and VBOs".format(self.name))
        self.vao = self._make_gl_vao()
        self.ind_vbo = self._make_gl_index_buffer(self.indexes)
        self.coord_vbo = self._make_gl_coord_buffer(self.vm_object.frames[0], self.shader_program)
        colors = self.vm_session.vm_config.gl_parameters["picking_dots_color"] * self.vm_object.frames.shape[1]
        colors = np.array(colors, dtype=np.float32).reshape([self.vm_object.frames.shape[1], 3])
        self.col_vbo = self._make_gl_color_buffer(colors, self.shader_program)
    
    def draw_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        _size = self.vm_session.vm_config.gl_parameters["dot_sel_size"]
        _size = _size * self.vm_glcore.height / (abs(self.vm_glcore.dist_cam_zrp)) / 2 
        

        #How to pass data to shader on the fly. It's not the most efficient mode, but it's an okay solution in this case.
        '''
        Using the "GL.glPointSize(_size)" function did not give 
        consistent results between different versions of openGL, 
        or linux distros. The solution, for now, was to pass the 
        pixel size value directly to the shader code.
        '''
        GL.glUseProgram(self.shader_program)
        custom_int_location = GL.glGetUniformLocation(self.shader_program, "_size")
        if custom_int_location != -1:
            GL.glUniform1i(custom_int_location, int(_size))  # Set the integer value to  _size
        else:
            #print("Uniform '_size' not found in shader program.")
            #print(int(_size))
            #GL.glPointSize(int(_size+256))
            GL.glPointSize(int(_size))
        
        GL.glEnable(GL.GL_VERTEX_PROGRAM_POINT_SIZE)
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        GL.glBindVertexArray(self.vao)
        
        if self.was_rep_coord_modified:
            self._load_coord_vbo(coord_vbo=True)
            self.was_rep_coord_modified = False
        if self.was_rep_ind_modified:
            self._load_ind_vbo(ind_vbo=True)
            self.was_rep_ind_modified = False
        
        GL.glDrawElements(GL.GL_POINTS, self.elements, GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_VERTEX_PROGRAM_POINT_SIZE)
        #GL.glPointSize(1)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
    
    def draw_background_sel_representation(self):
        pass


class DotsRepresentation(Representation):
    """ Class doc """
    
    def __init__ (self, vismol_object, vismol_glcore, indexes, active=True):
        """ Class initialiser """
        super(DotsRepresentation, self).__init__(vismol_object, vismol_glcore, "dots", active, indexes)
    
    def draw_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        self._enable_anti_alias_to_lines()
        _size = self.vm_glcore.vm_config.gl_parameters["dots_size"]
        _height = self.vm_glcore.height
        GL.glEnable(GL.GL_VERTEX_PROGRAM_POINT_SIZE)
        GL.glUseProgram(self.shader_program)
        GL.glPointSize(_size * _height / (abs(self.vm_glcore.dist_cam_zrp)) / 2)
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_fog(self.shader_program)
        self.vm_glcore.load_lights(self.shader_program)
        GL.glBindVertexArray(self.vao)
        
        if self.was_rep_coord_modified:
            self._load_coord_vbo(coord_vbo=True)
            self.was_rep_coord_modified = False
        if self.was_rep_ind_modified:
            self._load_ind_vbo(ind_vbo=True)
            self.was_rep_ind_modified = False
        if self.was_col_modified:
            self._load_color_vbo(None)
            self.was_col_modified = False
        
        GL.glDrawElements(GL.GL_POINTS, self.elements, GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        self._disable_anti_alias_to_lines()
        GL.glDisable(GL.GL_VERTEX_PROGRAM_POINT_SIZE)
        GL.glPointSize(1)
        GL.glUseProgram(0)
    
    def draw_background_sel_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        _size = self.vm_glcore.vm_config.gl_parameters["dots_size"]
        _height = self.vm_glcore.height
        GL.glEnable(GL.GL_VERTEX_PROGRAM_POINT_SIZE)
        GL.glUseProgram(self.sel_shader_program)
        GL.glPointSize(_size * _height / (abs(self.vm_glcore.dist_cam_zrp)) / 2)
        self.vm_glcore.load_matrices(self.sel_shader_program, self.vm_object.model_mat)
        GL.glBindVertexArray(self.sel_vao)
        
        if self.was_sel_coord_modified:
            self._load_coord_vbo(sel_coord_vbo=True)
            self.was_sel_coord_modified = False
        if self.was_sel_ind_modified:
            self._load_ind_vbo(sel_ind_vbo=True)
            self.was_sel_ind_modified = False
        
        GL.glDrawElements(GL.GL_POINTS, self.elements, GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glPointSize(1)
        GL.glUseProgram(0)


class LinesRepresentation(Representation):
    """ Class doc """
    
    def __init__(self, vismol_object, vismol_glcore, indexes, active=True):
        """ Class initialiser """
        super(LinesRepresentation, self).__init__(vismol_object, vismol_glcore, "lines", active, indexes)
        
        #self.elements = np.uint32(self.indexes.shape[0])
        self.size_vbo  =  None
        self.ratio_vbo =  None
        #print(self.vm_object.cov_radii_array)

    def draw_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        self._enable_anti_alias_to_lines()
        GL.glUseProgram(self.shader_program)
        line_width = self.vm_session.vm_config.gl_parameters["line_width"]
        line_width = (line_width*200/abs(self.vm_glcore.dist_cam_zrp)/2)**0.5
        GL.glLineWidth(line_width)
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_fog(self.shader_program)
        GL.glBindVertexArray(self.vao)
        #print(self.vm_object.cov_radii_array)
        
        '''simples bonds  or multiple bonds'''
        if self.ratio_vbo == None:
            try:
                self.vm_object._get_covalent_radii()
                self.size_vbo, self.ratio_vbo = self._make_gl_impostor_buffer(self.vm_object.cov_radii_array, self.shader_program)
            except:
                pass
                #print('Failed: self.vm_object._get_covalent_radii()')
        else:
            pass
            #self.size_vbo, self.ratio_vbo = self._make_gl_impostor_buffer(self.vm_object.cov_radii_array, self.shader_program)



        if self.was_rep_coord_modified:
            self._load_coord_vbo(coord_vbo=True)
            self.was_rep_coord_modified = False
            
            #
        if self.was_rep_ind_modified:
            self._load_ind_vbo(ind_vbo=True)
            self.was_rep_ind_modified = False
        if self.was_col_modified:
            self._load_color_vbo(None)
            self.was_col_modified = False
        
        GL.glDrawElements(GL.GL_LINES, self.elements, GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        self._disable_anti_alias_to_lines()
        GL.glLineWidth(1)
        GL.glUseProgram(0)

    def draw_background_sel_representation(self, line_width_factor=5):
        """ Function doc """
        self._check_vao_and_vbos()
        self._disable_anti_alias_to_lines()
        line_width = self.vm_session.vm_config.gl_parameters["line_width_selection"]
        GL.glUseProgram(self.sel_shader_program)
        GL.glLineWidth(line_width) # line_width_factor -> turn the lines thicker
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.vm_glcore.load_matrices(self.sel_shader_program, self.vm_object.model_mat)
        GL.glBindVertexArray(self.sel_vao)
        
        if self.was_sel_coord_modified:
            self._load_coord_vbo(sel_coord_vbo=True)
            self.was_sel_coord_modified = False
        if self.was_sel_ind_modified:
            self._load_ind_vbo(sel_ind_vbo=True)
            self.was_sel_ind_modified = False
        
        GL.glDrawElements(GL.GL_LINES, self.elements, GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(1)
        GL.glUseProgram(0)


class NonBondedRepresentation(Representation):
    """ Class doc """
    
    def __init__ (self, vismol_object, vismol_glcore, indexes, active=True):
        """ Class initialiser """
        super(NonBondedRepresentation, self).__init__(vismol_object, vismol_glcore, "nonbonded", active, indexes)
    
    def draw_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        self._enable_anti_alias_to_lines()
        line_width = self.vm_session.vm_config.gl_parameters["line_width"]
        GL.glUseProgram(self.shader_program)
        GL.glLineWidth(line_width*20/abs(self.vm_glcore.dist_cam_zrp))
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_fog(self.shader_program)
        GL.glBindVertexArray(self.vao)
        
        if self.was_rep_coord_modified:
            self._load_coord_vbo(coord_vbo=True)
            self.was_rep_coord_modified = False
        if self.was_rep_ind_modified:
            self._load_ind_vbo(ind_vbo=True)
            self.was_rep_ind_modified = False
        if self.was_col_modified:
            self._load_color_vbo(None)
            self.was_col_modified = False
        
        GL.glDrawElements(GL.GL_POINTS, self.elements, GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        self._disable_anti_alias_to_lines()
        GL.glLineWidth(1)
        GL.glUseProgram(0)
    
    def draw_background_sel_representation(self, line_width_factor=5):
        """ Function doc """
        self._check_vao_and_vbos()
        self._disable_anti_alias_to_lines()
        GL.glUseProgram(self.sel_shader_program)
        GL.glLineWidth(line_width_factor)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.vm_glcore.load_matrices(self.sel_shader_program, self.vm_object.model_mat)
        GL.glBindVertexArray(self.sel_vao)
        
        if self.was_sel_coord_modified:
            self._load_coord_vbo(sel_coord_vbo=True)
            self.was_sel_coord_modified = False
        if self.was_sel_ind_modified:
            self._load_ind_vbo(sel_ind_vbo=True)
            self.was_sel_ind_modified = False
        
        GL.glDrawElements(GL.GL_POINTS, self.elements, GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(1)
        GL.glUseProgram(0)


class SticksRepresentation(Representation):
    """ Class doc """
    
    def __init__(self, vismol_object, vismol_glcore, indexes, active=True, is_dynamic = False, name = "sticks"):
        """ Class initialiser """
        super(SticksRepresentation, self).__init__(vismol_object, vismol_glcore, name, active, indexes, is_dynamic)
        if  name == "sticks":
            self.radius = self.vm_session.vm_config.gl_parameters["sticks_radius"]
            self.spheres = None
        else:
            self.radius = self.vm_session.vm_config.gl_parameters["ribbon_width"]
            #print(set(indexes))
            self.spheres = SpheresRepresentation(vismol_object, vismol_glcore,
                                            active=True, indexes=list(vismol_object.atoms.keys()))
            

    def set_radius (self, radius):
        """ Function doc """
        self.radius = radius

    def _load_camera_pos(self, program):
        xyz_coords = self.vm_glcore.glcamera.get_modelview_position(self.vm_object.model_mat)
        u_campos = GL.glGetUniformLocation(program, "u_campos")
        GL.glUniform3fv(u_campos, 1, xyz_coords)
    
    def draw_representation(self):
        """ Function doc """
        self._check_vao_and_vbos ()
        self._enable_anti_alias_to_lines()
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glUseProgram(self.shader_program)
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_fog(self.shader_program)
        self.vm_glcore.load_lights(self.shader_program)
        self._load_camera_pos(self.shader_program)
        GL.glBindVertexArray(self.vao)
        
        #radius = self.vm_session.vm_config.gl_parameters["sticks_radius"]
        custom_int_location = GL.glGetUniformLocation(self.shader_program, "vert_rad")
        GL.glUniform1f(custom_int_location, self.radius)  # Set the integer value to  _size
        
        
        if self.was_rep_coord_modified:
            self._load_coord_vbo(coord_vbo=True)
            self.was_rep_coord_modified = False
        if self.was_rep_ind_modified:
            self._load_ind_vbo(ind_vbo=True)
            self.was_rep_ind_modified = False
        if self.was_col_modified:
            self._load_color_vbo(None)
            self.was_col_modified = False
        
        #GL.glDrawElements(GL.GL_LINES, int(len(self.vm_object.index_bonds)*2), GL.GL_UNSIGNED_INT, None)
        if self.is_dynamic:
            # it's working  fine for QC and QC/MM but, 
            # I had to set 'int(len(self.vm_object.index_bonds)+4)' once some bonds wasn't 
            # properly appearing on the glArea
            GL.glDrawElements(GL.GL_LINES, int(len(self.vm_object.index_bonds)+4), GL.GL_UNSIGNED_INT, None)
        else:
            #normal rep
            GL.glDrawElements(GL.GL_LINES, int(len(self.vm_object.index_bonds)), GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        self._disable_anti_alias_to_lines()
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glUseProgram(0)
        GL.glLineWidth(1)
        #print(self.spheres)
        #if self.spheres:
        #    self.spheres.draw_representation()
        #    print(self.spheres.sphere_indexes)
            
    def draw_background_sel_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        self._disable_anti_alias_to_lines()
        GL.glUseProgram(self.sel_shader_program)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.vm_glcore.load_matrices(self.sel_shader_program, self.vm_object.model_mat)
        GL.glBindVertexArray(self.sel_vao)
        
        #radius = self.vm_session.vm_config.gl_parameters["sticks_radius"]
        custom_int_location = GL.glGetUniformLocation(self.sel_shader_program, "vert_rad")
        GL.glUniform1f(custom_int_location, self.radius)
        
        
        if self.was_sel_coord_modified:
            self._load_coord_vbo(sel_coord_vbo=True)
            self.was_sel_coord_modified = False
        if self.was_sel_ind_modified:
            self._load_ind_vbo(sel_ind_vbo=True)
            self.was_sel_ind_modified = False
        
        GL.glDrawElements(GL.GL_LINES, self.elements, GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(1)
        GL.glUseProgram(0)


class SpheresRepresentation(Representation):
    """ Class doc """
    #            self, vismol_object, vismol_glcore, indexes, active=True, is_dynamic = False, name = "sticks"
    def __init__(self, vismol_object, vismol_glcore, indexes, active=True, vdw = False, mode = 0):
        """ Class initialiser """
        self.mode = mode
        if self.mode == 0 or self.mode == 2 or self.mode == 3:
            super(SpheresRepresentation, self).__init__(vismol_object, vismol_glcore, "spheres", active, indexes)
        elif self.mode ==1:
            super(SpheresRepresentation, self).__init__(vismol_object, vismol_glcore, "picking_spheres", active, indexes)
        
        
        else:
            pass
        import vismol.utils.sphere_data as sphd
        
        if self.mode == 2:
            #for ribbon spheres
            self.level = 1
            self.scale = 1
            self.rad   = 1.98
        
        if self.mode == 3:
            #for ribbon spheres
            self.level = 1
            self.scale = 1
            self.rad   = 0.16
        
        
        else:
            self.level = self.vm_session.vm_config.gl_parameters["sphere_quality"]
            self.scale = self.vm_session.vm_config.gl_parameters["sphere_scale"]
        
        if vdw:
            self.sphere_vertices = sphd.sphere_vertices[self.level] 
        else:
            self.sphere_vertices = sphd.sphere_vertices[self.level]* self.scale
        self.sphere_indexes = sphd.sphere_triangles[self.level]
        self.instances_elemns = self.sphere_indexes.shape[0]
    
    def _make_gl_representation_vao_and_vbos(self):
        """ Function doc """
        logger.debug("building '{}' representation VAO and VBOs".format(self.name))
        self.vao = self._make_gl_vao()
        self.ind_vbo = self._make_gl_index_buffer(self.sphere_indexes)
        self.coord_vbo = self._make_gl_coord_buffer(self.sphere_vertices, self.shader_program)
        self.col_vbo = self._make_gl_color_buffer(np.zeros(3, dtype=np.float32), self.shader_program, instances=True)
        self.rad_vbo = self._make_gl_radius_buffer(np.zeros(1, dtype=np.float32), self.shader_program, instances=True)
        self.insta_vbo = self._make_gl_instance_buffer(np.zeros(3, dtype=np.float32), self.shader_program)
    
    def _make_gl_sel_representation_vao_and_vbos(self):
        """ Function doc """
        logger.debug("building '{}' background selection VAO and VBOs".format(self.name))
        self.sel_vao = self._make_gl_vao()
        self.sel_ind_vbo = self._make_gl_index_buffer(self.sphere_indexes)
        self.sel_coord_vbo = self._make_gl_coord_buffer(self.sphere_vertices, self.sel_shader_program)
        self.sel_col_vbo = self._make_gl_color_buffer(np.zeros(3, dtype=np.float32), self.sel_shader_program, instances=True)
        self.sel_rad_vbo = self._make_gl_radius_buffer(np.zeros(1, dtype=np.float32), self.shader_program, instances=True)
        self.sel_insta_vbo = self._make_gl_instance_buffer(np.zeros(3, dtype=np.float32), self.shader_program)
    
    def _coords_colors_rads(self):
        coords, colors, rads = [], [], []
        frame, f = self.vm_glcore._safe_frame_coords(self.vm_object)
        for i in self.indexes:
            coords.append(frame[i])
            colors.append(self.vm_object.atoms[i].color)
            if self.mode == 2 or self.mode == 3:
                rads.append(self.rad)
            else:
                rads.append(self.vm_object.atoms[i].ball_rad)
        coords = np.array(coords, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        rads = np.array(rads, dtype=np.float32)
        return coords, colors, rads
    
    def _sel_coords_colors_rads(self):
        coords, colors, rads = [], [], []
        frame, f = self.vm_glcore._safe_frame_coords(self.vm_object)
        for i in self.indexes:
            coords.append(frame[i])
            colors.append(self.vm_object.atoms[i].color_id)
            rads.append(self.vm_object.atoms[i].ball_rad)
        coords = np.array(coords, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        rads = np.array(rads, dtype=np.float32)
        return coords, colors, rads
    
    def draw_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_BLEND)
        GL.glUseProgram(self.shader_program)
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_lights(self.shader_program)
        self.vm_glcore.load_fog(self.shader_program)
        GL.glBindVertexArray(self.vao)
        
        if self.was_rep_coord_modified or self.was_rep_ind_modified:
            coords, colors, rads = self._coords_colors_rads()
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.insta_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, coords.nbytes, coords, GL.GL_STATIC_DRAW)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.col_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.nbytes, colors, GL.GL_STATIC_DRAW)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.rad_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, rads.nbytes, rads, GL.GL_STATIC_DRAW)
            self.elements = np.uint32(coords.shape[0])
            self.was_rep_coord_modified = False
            self.was_rep_ind_modified = False
        
        GL.glDrawElementsInstanced(GL.GL_TRIANGLES, self.instances_elemns, GL.GL_UNSIGNED_INT, None, self.elements)
        
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        #print('aqui', self.sphere_indexes)

    def draw_background_sel_representation(self):
        """ Function doc """
        
        if self.mode == 2 or self.mode == 3:
            #Selection is not necessary
            return None
        
        self._check_vao_and_vbos()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self.sel_shader_program)
        self.vm_glcore.load_matrices(self.sel_shader_program, self.vm_object.model_mat)
        GL.glBindVertexArray(self.sel_vao)
        
        if self.was_sel_coord_modified or self.was_sel_ind_modified:
            coords, colors, rads = self._sel_coords_colors_rads()
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.sel_insta_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, coords.nbytes, coords, GL.GL_STATIC_DRAW)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.sel_col_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.nbytes, colors, GL.GL_STATIC_DRAW)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.sel_rad_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, rads.nbytes, rads, GL.GL_STATIC_DRAW)
            self.elements = np.uint32(coords.shape[0])
            self.was_sel_coord_modified = False
            self.was_sel_ind_modified = False
        
        GL.glDrawElementsInstanced(GL.GL_TRIANGLES, self.instances_elemns, GL.GL_UNSIGNED_INT, None, self.elements)
        
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(0)


class DashedLinesRepresentation(Representation):
    """ Class doc """
    
    def __init__(self, vismol_object, vismol_glcore, indexes, active=True, depth_test = False):
        """ Class initialiser """
        super(DashedLinesRepresentation, self).__init__(vismol_object, vismol_glcore, "dash", active, indexes)
        self.depth_test = depth_test
        self.color2 = [1.0 ,1.0, 0.0]
    
    def draw_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        self._enable_anti_alias_to_lines()
        GL.glUseProgram(self.shader_program)
        line_width = self.vm_session.vm_config.gl_parameters["line_width"]
        line_width = (line_width*200/abs(self.vm_glcore.dist_cam_zrp)/2)**0.5
        GL.glLineWidth(line_width)
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_fog(self.shader_program)
        GL.glBindVertexArray(self.vao)
        
        # How to pass data to shader on the fly. It's not the most 
        # efficient mode, but it's an okay solution in this case.
        color = GL.glGetUniformLocation(self.shader_program, "uniform_color")
        #GL.glUniformMatrix4fv(proj, 1, GL.GL_FALSE, self.glcamera.projection_matrix)
        color2 = np.array(self.color2, dtype=np.float32)
        GL.glUniform3fv(color, 1, color2)
        
        '''
        #How to pass data to shader on the fly. It's not the most efficient wood, but it's an okay solution in this case.
        test = GL.glGetUniformLocation(self.shader_program, "test_int")
        test_int = 5
        GL.glUniform1i(test, test_int)
        '''
        
        if self.depth_test:
            pass
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)
        
        
        
        if self.was_rep_coord_modified:
            self._load_coord_vbo(coord_vbo=True)
            self.was_rep_coord_modified = False
        if self.was_rep_ind_modified:
            self._load_ind_vbo(ind_vbo=True)
            self.was_rep_ind_modified = False
        if self.was_col_modified:
            self._load_color_vbo(None)
            self.was_col_modified = False
        
        GL.glDrawElements(GL.GL_LINES, self.elements, GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        self._disable_anti_alias_to_lines()
        GL.glLineWidth(1)
        GL.glUseProgram(0)
        
    def draw_background_sel_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        self._disable_anti_alias_to_lines()
        line_width = self.vm_session.vm_config.gl_parameters["line_width_selection"]
        GL.glUseProgram(self.sel_shader_program)
        GL.glLineWidth(line_width)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.vm_glcore.load_matrices(self.sel_shader_program, self.vm_object.model_mat)
        GL.glBindVertexArray(self.sel_vao)
        
        if self.was_sel_coord_modified:
            self._load_coord_vbo(sel_coord_vbo=True)
            self.was_sel_coord_modified = False
        if self.was_sel_ind_modified:
            self._load_ind_vbo(sel_ind_vbo=True)
            self.was_sel_ind_modified = False
        
        GL.glDrawElements(GL.GL_LINES, self.elements, GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(1)
        GL.glUseProgram(0)


class ImpostorRepresentation(Representation):
    """ Class doc """
    
    def __init__ (self, vismol_object, vismol_glcore, indexes, active=True):
        """ Class initialiser """
        super(ImpostorRepresentation, self).__init__(vismol_object, vismol_glcore, "impostor", active, indexes)
    
    def _make_gl_representation_vao_and_vbos(self):
        """ Function doc """
        logger.debug("building '{}' representation VAO and VBOs".format(self.name))
        self.vao = self._make_gl_vao()
        self.ind_vbo = self._make_gl_index_buffer(self.indexes)
        self.coord_vbo = self._make_gl_coord_buffer(self.vm_object.frames[0], self.shader_program)
        self.col_vbo = self._make_gl_color_buffer(self.vm_object.colors, self.shader_program)
        self.size_vbo, self.ratio_vbo = self._make_gl_impostor_buffer(self.vm_object.cov_radii_array, self.shader_program)
    
    def _make_gl_sel_representation_vao_and_vbos(self):
        """ Function doc """
        logger.debug("building '{}' background selection VAO and VBOs".format(self.name))
        self.sel_vao = self._make_gl_vao()
        self.sel_ind_vbo = self._make_gl_index_buffer(self.indexes)
        self.sel_coord_vbo = self._make_gl_coord_buffer(self.vm_object.frames[0], self.sel_shader_program)
        self.sel_col_vbo = self._make_gl_color_buffer(self.vm_object.color_indexes, self.sel_shader_program)
        self.sel_size_vbo, self.sel_ratio_vbo = self._make_gl_impostor_buffer(self.vm_object.cov_radii_array, self.shader_program)
    
    # def _modified_window_size(self):
    #     ratio = self.vm_glcore.width / self.vm_glcore.height
    #     if self.ratio != ratio:
    #         return True
    #     return False
    
    def _load_camera_pos(self, program):
        xyz_coords = self.vm_glcore.glcamera.get_modelview_position(self.vm_object.model_mat)
        u_campos = GL.glGetUniformLocation(program, "u_campos")
        GL.glUniform3fv(u_campos, 1, xyz_coords)
    
    def draw_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        # if self._modified_window_size():
        #     self._load_impostor_ratio_vbo(coord_vbo=True)
        self._enable_anti_alias_to_lines()
        GL.glUseProgram(self.shader_program)
        GL.glEnable(GL.GL_VERTEX_PROGRAM_POINT_SIZE)
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_fog(self.shader_program)
        self.vm_glcore.load_lights(self.shader_program)
        self._load_camera_pos(self.shader_program)
        GL.glBindVertexArray(self.vao)
        # pm = self.vm_glcore.glcamera.projection_matrix
        # print(pm[0,0] * pm[1,1], "fov")
        
        if self.was_rep_coord_modified:
            self._load_coord_vbo(coord_vbo=True)
            self.was_rep_coord_modified = False
        if self.was_rep_ind_modified:
            self._load_ind_vbo(ind_vbo=True)
            self.was_rep_ind_modified = False
        if self.was_col_modified:
            self._load_color_vbo(None)
            self.was_col_modified = False
        
        GL.glDrawElements(GL.GL_POINTS, self.elements, GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        self._disable_anti_alias_to_lines()
        GL.glPointSize(1)
        GL.glUseProgram(0)
    
    def draw_background_sel_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        # if self._modified_window_size():
        #     self._load_impostor_ratio_vbo(sel_coord_vbo=True)
        GL.glUseProgram(self.sel_shader_program)
        GL.glEnable(GL.GL_VERTEX_PROGRAM_POINT_SIZE)
        self.vm_glcore.load_matrices(self.sel_shader_program, self.vm_object.model_mat)
        # self.vm_glcore.load_lights(self.shader_program)
        self._load_camera_pos(self.sel_shader_program)
        GL.glBindVertexArray(self.sel_vao)
        
        if self.was_sel_coord_modified:
            self._load_coord_vbo(sel_coord_vbo=True)
            self.was_sel_coord_modified = False
        if self.was_sel_ind_modified:
            self._load_ind_vbo(sel_ind_vbo=True)
            self.was_sel_ind_modified = False
        
        GL.glDrawElements(GL.GL_POINTS, self.elements, GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glPointSize(1)
        GL.glUseProgram(0)


class CellLineRepresentation:
    """ Class doc 
    
    This is a modified version of the Representations class. 
    The differences are in the methods:

    # _make_gl_representation_vao_and_vbos
    # _load_coord_vbo
    # _load_color_vbo

    In this case, there is a modification in the code so that 
    the coordinates, indices and colors transmitted to the GPU 
    refer to the cell parameters.

    In this class there is no selection presentation.
    """
    
    def __init__ (self, vismol_object, vismol_glcore, name, active, indexes, is_dynamic = False):
        self.vm_object = vismol_object
        self.vm_session = vismol_object.vm_session
        self.vm_glcore = vismol_glcore
        self.name = name
        self.active = active
        self.indexes = np.array(indexes, dtype=np.uint32)
        self.elements = np.uint32(self.indexes.shape[0])
        
        self.is_dynamic = is_dynamic
        
        self.was_rep_modified = False
        self.was_sel_modified = False
        self.was_col_modified = False
        self.was_rep_coord_modified = False
        self.was_sel_coord_modified = False
        self.was_rep_ind_modified = False
        self.was_sel_ind_modified = False
        self.was_rep_col_modified = False
        # representation
        self.vao = None
        self.ind_vbo = None
        self.coord_vbo = None
        self.col_vbo = None
        self.size_vbo = None
        # selection
        #self.sel_vao = None
        #self.sel_ind_vbo = None
        #self.sel_coord_vbo = None
        #self.sel_col_vbo = None
        #self.sel_size_vbo = None
        # shaders
        self.shader_program = None
        self.sel_shader_program = None
    
    def _check_vao_and_vbos(self):
        #print(self.name)
        self.shader_program = self.vm_glcore.shader_programs[self.name]
        #self.sel_shader_program = self.vm_glcore.shader_programs[self.name + "_sel"]
        if self.vao is None:
            self._make_gl_representation_vao_and_vbos()
        #if self.sel_vao is None:
        #    self._make_gl_sel_representation_vao_and_vbos()
    
    def _make_gl_representation_vao_and_vbos(self):
        """ Function doc """
        logger.debug("building '{}' representation VAO and VBOs".format(self.name))
        self.vao = self._make_gl_vao()
        self.ind_vbo   = self._make_gl_index_buffer(self.indexes)
        self.coord_vbo = self._make_gl_coord_buffer(self.vm_object.cell_coordinates[0], self.shader_program)
        self.col_vbo   = self._make_gl_color_buffer(self.vm_object.cell_colors, self.shader_program)
    
    #def _make_gl_sel_representation_vao_and_vbos(self):
    #    """ Function doc """
    #    logger.debug("building '{}' background selection VAO and VBOs".format(self.name))
    #    self.sel_vao = self._make_gl_vao()
    #    self.sel_ind_vbo = self._make_gl_index_buffer(self.indexes)
    #    self.sel_coord_vbo = self._make_gl_coord_buffer(self.vm_object.cell_coordinates[0], self.sel_shader_program)
    #    self.sel_col_vbo = self._make_gl_color_buffer(self.vm_object.color_indexes, self.sel_shader_program)
    
    def _make_gl_vao(self):
        """ Function doc """
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        return vao
    
    def _make_gl_index_buffer(self, indexes):
        """ Function doc """
        ind_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ind_vbo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indexes.nbytes, indexes, GL.GL_DYNAMIC_DRAW)
        return ind_vbo
    
    def _make_gl_coord_buffer(self, coords, program):
        """ Function doc """
        coord_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, coord_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, coords.nbytes, coords, GL.GL_STATIC_DRAW)
        att_position = GL.glGetAttribLocation(program, "vert_coord")
        GL.glEnableVertexAttribArray(att_position)
        GL.glVertexAttribPointer(att_position, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*coords.itemsize, ctypes.c_void_p(0))
        return coord_vbo
    
    def _make_gl_color_buffer(self, colors, program, instances=False):
        """ Function doc """
        col_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, col_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.nbytes, colors, GL.GL_STATIC_DRAW)
        att_colors = GL.glGetAttribLocation(program, "vert_color")
        GL.glEnableVertexAttribArray(att_colors)
        GL.glVertexAttribPointer(att_colors, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*colors.itemsize, ctypes.c_void_p(0))
        if instances:
            GL.glVertexAttribDivisor(att_colors, 1)
        return col_vbo
    
    #def _make_gl_radius_buffer(self, radii, program, instances=False):
    #    """ Function doc """
    #    rad_vbo = GL.glGenBuffers(1)
    #    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, rad_vbo)
    #    GL.glBufferData(GL.GL_ARRAY_BUFFER, radii.nbytes, radii, GL.GL_STATIC_DRAW)
    #    att_rads = GL.glGetAttribLocation(program, "vert_radius")
    #    GL.glEnableVertexAttribArray(att_rads)
    #    GL.glVertexAttribPointer(att_rads, 1, GL.GL_FLOAT, GL.GL_FALSE, radii.itemsize, ctypes.c_void_p(0))
    #    if instances:
    #        GL.glVertexAttribDivisor(att_rads, 1)
    #    return rad_vbo
    
    #def _make_gl_instance_buffer(self, instances, program):
    #    """ Function doc """
    #    insta_vbo = GL.glGenBuffers(1)
    #    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, insta_vbo)
    #    GL.glBufferData(GL.GL_ARRAY_BUFFER, instances.nbytes, instances, GL.GL_STATIC_DRAW)
    #    gl_insta = GL.glGetAttribLocation(program, "vert_instance")
    #    GL.glEnableVertexAttribArray(gl_insta)
    #    GL.glVertexAttribPointer(gl_insta, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
    #    GL.glVertexAttribDivisor(gl_insta, 1)
    #    return insta_vbo
    
    #def _make_gl_impostor_buffer(self, impostors_radii, program):
    #    """ Function doc """
    #    size_vbo = GL.glGenBuffers(1)
    #    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, size_vbo)
    #    GL.glBufferData(GL.GL_ARRAY_BUFFER, impostors_radii.nbytes, impostors_radii, GL.GL_STATIC_DRAW)
    #    att_size = GL.glGetAttribLocation(program, "vert_dot_size")
    #    GL.glEnableVertexAttribArray(att_size)
    #    GL.glVertexAttribPointer(att_size, 1, GL.GL_FLOAT, GL.GL_FALSE, impostors_radii.itemsize, ctypes.c_void_p(0))
    #    
    #    self.ratio = self.vm_glcore.width / self.vm_glcore.height
    #    ratio_vbo = 1
    #    # ratio = np.repeat(self.ratio, impostors_radii.shape[0])
    #    # ratio_vbo = GL.glGenBuffers(1)
    #    # GL.glBindBuffer(GL.GL_ARRAY_BUFFER, ratio_vbo)
    #    # GL.glBufferData(GL.GL_ARRAY_BUFFER, ratio.nbytes, ratio, GL.GL_STATIC_DRAW)
    #    # att_ratio = GL.glGetAttribLocation(program, "hw_ratio")
    #    # GL.glEnableVertexAttribArray(att_ratio)
    #    # GL.glVertexAttribPointer(att_ratio, 1, GL.GL_FLOAT, GL.GL_FALSE, ratio.itemsize, ctypes.c_void_p(0))
    #    return size_vbo, ratio_vbo
    
    def _load_coord_vbo(self, coord_vbo=False, sel_coord_vbo=False):
        """ This function assigns the coordinates to 
        be drawn by the function  draw_representation"""
        #frame, f = self.vm_glcore._safe_frame_coords(self.vm_object)
        frame = self.vm_object.cell_coordinates[0]
        if coord_vbo:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.coord_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, frame.nbytes, frame, GL.GL_STATIC_DRAW)
        
        #if sel_coord_vbo:
        #    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.sel_coord_vbo)
        #    GL.glBufferData(GL.GL_ARRAY_BUFFER, frame.nbytes, frame, GL.GL_STATIC_DRAW)
    
    def _load_ind_vbo(self, ind_vbo=False, sel_ind_vbo=False):
        """ Function doc """
        #if self.is_dynamic:
        #    frame, f = self.vm_glcore._safe_frame_coords(self.vm_object)
        #    self.define_new_indexes_to_vbo(input_indexes = self.vm_object.dynamic_bonds[f])
            
        if ind_vbo:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ind_vbo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.indexes.nbytes, self.indexes, GL.GL_DYNAMIC_DRAW)
        
        #if sel_ind_vbo:
        #    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.sel_ind_vbo)
        #    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.indexes.nbytes, self.indexes, GL.GL_DYNAMIC_DRAW)
    
    def _load_color_vbo(self, colors):
        """ This function assigns the colors to
            be drawn by the function  draw_representation"""
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.col_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.vm_object.cell_colors.nbytes, self.vm_object.cell_colors, GL.GL_STATIC_DRAW)
    
    def _enable_anti_alias_to_lines(self):
        """ Function doc """
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
    
    def _disable_anti_alias_to_lines(self):
        """ Function doc """
        GL.glDisable(GL.GL_LINE_SMOOTH)
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_DEPTH_TEST)
    
    def define_new_indexes_to_vbo(self, input_indexes):
        """ Function doc """
        self.indexes = np.array(input_indexes, dtype=np.uint32)
        self.elements = np.uint32(self.indexes.shape[0])


    def draw_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        self._enable_anti_alias_to_lines()
        GL.glUseProgram(self.shader_program)
        line_width = self.vm_session.vm_config.gl_parameters["line_width"]
        line_width = (line_width*200/abs(self.vm_glcore.dist_cam_zrp)/2)**0.5
        GL.glLineWidth(line_width)
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_fog(self.shader_program)
        GL.glBindVertexArray(self.vao)
        
        if self.was_rep_coord_modified:
            self._load_coord_vbo(coord_vbo=True)
            self.was_rep_coord_modified = False
        if self.was_rep_ind_modified:
            self._load_ind_vbo(ind_vbo=True)
            self.was_rep_ind_modified = False
        #self.was_col_modified = True
        if self.was_col_modified:
            self._load_color_vbo(None)
            self.was_col_modified = False
        
        GL.glDrawElements(GL.GL_LINES, self.elements, GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        self._disable_anti_alias_to_lines()
        GL.glLineWidth(1)
        GL.glUseProgram(0)

    def draw_background_sel_representation(self, line_width_factor=5):
        """ Function doc """
        pass


class LabelRepresentation:
    
    def __init__ (self, vismol_object, vismol_glcore, indexes, labels, color = [1, 1, 1, 1]):
        self.vm_object = vismol_object
        self.vm_session = vismol_object.vm_session
        self.vm_glcore = vismol_glcore
        self.vm_font = VismolFont(color=color)
        self.indexes = indexes
        self.labels = labels
        self.active = True
        self.was_rep_ind_modified = False
        self.was_sel_ind_modified = False    
        self.is_dynamic = False
    def define_new_indexes_to_vbo(self, indexes):
        """ Function doc """
        self.indexes = indexes
    
    def draw_representation(self):
        """ Function doc """
        if self.vm_glcore.dragging:
            return False

        if self.vm_font.vao is None:
            self.vm_font.set_dimensions (width = 0.15, height= 0.18 )
            #self.vm_font.set_color(r = 255, g = 0, b =0)
            self.vm_font.make_freetype_font()
            #self.vm_font.make_freetype_texture(self.core_shader_programs["freetype"])
            self.vm_font.make_freetype_texture(self.vm_glcore.core_shader_programs["freetype"])
        
        
        
        number = 1
        self.chars = 0
        xyz_pos = []
        uv_coords = []
        
        
        #for vm_object in self.vm_session.vm_objects_dic.values():
        #for index, atom in vm_object.atoms.items():
        for index in self.indexes:
            atom = self.vm_object.atoms[index]
            
            
            
            text = atom.label_text#+'/'+str(atom.index)
            #text = atom.residue.name +'/'+ atom.name+'/'+str(atom.index)
            
            
            
            frame = self.vm_glcore._get_vismol_object_frame(atom.vm_object)
            x, y, z = atom.coords(frame)
            point = np.array([x, y, z, 1], dtype=np.float32)
            point = np.dot(point, self.vm_glcore.model_mat)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.vm_font.texture_id)
            for i, c in enumerate(text):
                self.chars += 1
                c_id = ord(c)
                x = c_id %  16      #  16  
                y = c_id // 16 - 2  #  16 - 2
                xyz_pos.append((point[0] + i * self.vm_font.char_width) - (len(text)*0.1)/2)
                xyz_pos.append(point[1])
                xyz_pos.append(point[2])
                uv_coords.append(x * self.vm_font.text_u)
                uv_coords.append(y * self.vm_font.text_v)
                uv_coords.append((x + 1) * self.vm_font.text_u)
                uv_coords.append((y + 1) * self.vm_font.text_v)



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
        GL.glUseProgram(self.vm_glcore.core_shader_programs["freetype"])
        self.do_once = False
        
        self.vm_font.load_matrices(self.vm_glcore.core_shader_programs["freetype"],
                                   self.vm_glcore.glcamera.view_matrix,
                                   self.vm_glcore.glcamera.projection_matrix)
        self.vm_font.load_font_params(self.vm_glcore.core_shader_programs["freetype"])
        
        GL.glBindVertexArray(self.vm_font.vao)
        GL.glDrawArrays(GL.GL_POINTS, 0, self.chars)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)


    def draw_background_sel_representation(self, line_width_factor=5):
        """ Function doc """
        pass


class CartoonRepresentation(Representation):
    def __init__ (self, name = 'cartoon', active = True, rep_type = 'mol', vismol_object = None, vismol_glcore = None, indexes = []):
        self.name               = name
        self.active             = active
        self.type               = rep_type

        self.vm_object             = vismol_object
        self.vm_glcore             = vm_glcore
        
        # representation 	
        self.vao            = None
        self.ind_vbo        = None
        self.coord_vbo      = None
        self.norm_vbo       = None
        self.col_vbo        = None
        self.size_vbo       = None
           

        # bgrd selection   
        self.sel_vao        = None
        self.sel_ind_vbo    = None
        self.sel_coord_vbo  = None
        self.sel_col_vbo    = None
        self.sel_size_vbo   = None


        #     S H A D E R S
        self.shader_program     = None
        self.sel_shader_program = None
        
        
        coords, normals, indexes, colors = cartoon.cartoon(vismol_object, spline_detail=5)
        
        coords = coords.flatten()
        normals = normals.flatten()
        colors = colors.flatten()
        
        
        self.coords2 = coords
        self.colors2 = colors
        self.normals2 = normals
        self.indexes2 = indexes


    def _make_gl_vao_and_vbos (self, indexes = None):
        """ Function doc """
        #if indexes is not None:
        #    pass
        #else:
        
        #dot_qtty  = int(len(self.vm_object.frames[0])/3)
        #indexes = []
        #for i in range(dot_qtty):
        #    indexes.append(i)
        

        self.shader_program     = self.vm_glcore.shader_programs[self.name]
        #self.sel_shader_program = self.vm_glcore.shader_programs[self.name+'_sel']
        

        """
        coords  = np.array(self.coords2, dtype=np.float32)
        colors  = np.array(self.colors2, dtype=np.float32)
        normals = np.array(self.normals2, dtype=np.float32)
        indexes = np.array(self.indexes2, dtype=np.uint32)
        """
        
        
        coords  = self.coords2 
        colors  = self.colors2 
        normals = self.normals2
        indexes = self.indexes2
        
        #print ('len(coords),len(colors), len(normals),len(indexes)', len(coords),len(colors), len(normals),len(indexes)  )

        self._make_gl_representation_vao_and_vbos (indexes    = indexes,
                                                   coords     = coords ,
                                                   colors     = colors ,
                                                   dot_sizes  = None   ,
                                                   normals    = normals
                                                   )
        
        
        
        self.ind_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ind_vbo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indexes.itemsize*len(indexes), indexes, GL.GL_DYNAMIC_DRAW)
        
        #self.coord_vbo = GL.glGenBuffers(1)
        #GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.coord_vbo)
        ##GL.glBufferData(GL.GL_ARRAY_BUFFER, coords.itemsize*len(coords), coords, GL.GL_STATIC_DRAW)
        #GL.glBufferData(GL.GL_ARRAY_BUFFER, coords.nbytes, coords, GL.GL_STATIC_DRAW)
        #gl_coord = GL.glGetAttribLocation(self.shader_program, 'vert_coord')
        #GL.glEnableVertexAttribArray(gl_coord)
        #GL.glVertexAttribPointer(gl_coord, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*coords.itemsize, ctypes.c_void_p(0))
        
        
        self.col_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.col_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.itemsize*len(colors), colors, GL.GL_STATIC_DRAW)
        gl_color = GL.glGetAttribLocation(self.shader_program, 'vert_color')
        GL.glEnableVertexAttribArray(gl_color)
        GL.glVertexAttribPointer(gl_color, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*colors.itemsize, ctypes.c_void_p(0))

        self.norm_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.norm_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, normals.itemsize*len(normals), normals, GL.GL_STATIC_DRAW)
        gl_norm = GL.glGetAttribLocation(self.shader_program, 'vert_norm')
        GL.glEnableVertexAttribArray(gl_norm)
        GL.glVertexAttribPointer(gl_norm, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*normals.itemsize, ctypes.c_void_p(0))
        
        
        
        
        
        #self.centr_vbo = GL.glGenBuffers(1)
        #GL.glBindBuffer(GL.GL_ARRAY_BUFFER, coords)
        #GL.glBufferData(GL.GL_ARRAY_BUFFER, coords.itemsize*len(coords), coords, GL.GL_STATIC_DRAW)
        #gl_center = GL.glGetAttribLocation(self.shader_program , 'vert_centr')
        #GL.glEnableVertexAttribArray(gl_center)
        #GL.glVertexAttribPointer(gl_center, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*coords.itemsize, ctypes.c_void_p(0))
        
        
        
        colors_idx = self.vm_object.color_indexes
        self.sel_vao = True
        """
        self._make_gl_sel_representation_vao_and_vbos (indexes    = indexes    ,
                                                       coords     = coords     ,
                                                       colors     = colors_idx ,
                                                       dot_sizes  = None       ,
                                                       )
        """
    def draw_representation (self):
        """ Function doc """
        self._check_vao_and_vbos ()
        #self._enable_anti_alias_to_lines()
        
        
        
        
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        #GL.glCullFace(GL.GL_BACK)
        view = self.vm_glcore.glcamera.view_matrix
        
        GL.glUseProgram(self.shader_program )
        
        #print (self.vm_object.model_mat,view)
        
        m_normal = np.array(np.matrix(np.dot(view, self.vm_object.model_mat)).I.T)
        
        self.vm_glcore.load_matrices(self.shader_program , self.vm_object.model_mat)
        self.vm_glcore.load_lights  (self.shader_program )
        self.vm_glcore.load_fog     (self.shader_program )
        GL.glBindVertexArray(self.vao)
        
        
        
        
        
        
        
        
        
        """
        #print ("DotsRepresentation")
        height = self.vm_glcore.height
        
        GL.glUseProgram(self.shader_program)
        #1*self.height dot_size
        #GL.glLineWidth(40/abs(self.vm_glcore.dist_cam_zrp))
        GL.glPointSize(0.1*height/abs(self.vm_glcore.dist_cam_zrp)) # dot size not included yet
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_fog(self.shader_program)
        GL.glBindVertexArray(self.vao)
        """
        if self.vm_glcore.modified_view:
            pass
        
        else:
            """
            This function checks if the number of the called frame will not exceed 
            the limit of frames that each object has. Allowing two objects with 
            different trajectory sizes to be manipulated at the same time within the 
            glArea"""
            # self._set_coordinates_to_buffer(coord_vbo = True, sel_coord_vbo = False)
            #GL.glDrawElements(GL.GL_POINTS, int(len(self.indexes2)), GL.GL_UNSIGNED_INT, None)
            #GL.glDrawElements(GL.GL_LINE_LOOP, int(len(self.coords2)), GL.GL_UNSIGNED_INT, None)
            #GL.glDrawElements(GL.GL_LINE_STRIP, int(len(self.indexes2)), GL.GL_UNSIGNED_INT, None)
            
            #print("int(len(self.indexes2))", int(len(self.indexes2)))
            GL.glDrawElements(GL.GL_TRIANGLES, int(len(self.indexes2)), GL.GL_UNSIGNED_INT, None)
            #GL.glDrawElements(GL.GL_TRIANGLES, 54060, GL.GL_UNSIGNED_INT, None)
        
        #GL.glBindVertexArray(0)
        #GL.glLineWidth(1)
        #GL.glUseProgram(0)
        #GL.glDisable(GL.GL_LINE_SMOOTH)
        #GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_DEPTH_TEST)
        
            
    def draw_background_sel_representation  (self):
        """ Function doc """
        pass


class SurfaceRepresentation(Representation):
    """ Class doc """
    

    def __init__(self, vismol_object, vismol_glcore, name, active, indexes, is_dynamic = False, iso_color = [0,1,1], surface_name = None):
    #def __init__(self, vismol_object = None, vismol_glcore = None, indexes = None, active=True, vdw = False, mode = 0):
        super(SurfaceRepresentation, self).__init__(vismol_object, vismol_glcore, name, active , indexes, is_dynamic)


        self.active    = True
        self.vm_glcore = vismol_glcore
        self.name      =  name   
        self.vismol_object = vismol_object
        self.surf_name     = surface_name

    def _make_gl_representation_vao_and_vbos(self, debug = False):
        """ Function doc """
        if debug:
            logger.debug("building '{}' representation VAO and VBOs".format(self.name))
        self.vao       = self._make_gl_vao()
        self.ind_vbo   = self._make_gl_index_buffer(self.indexes)
        self.coord_vbo = self._make_gl_coord_buffer(self.vertices, self.shader_program)
        #self.col_vbo   = self._make_gl_color_buffer(np.zeros(3, dtype=np.float32), self.shader_program, instances=True)
        self.col_vbo   = self._make_gl_color_buffer(self.colors, self.shader_program, instances=True)
    
    def _load_coord_vbo(self, coord_vbo=False, vertices = None):
        """ This function assigns the coordinates to 
        be drawn by the function  draw_representation"""
        frame, f = self.vm_glcore._safe_frame_coords(self.vm_object)
        if coord_vbo:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.coord_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

    def _load_ind_vbo(self, ind_vbo=False, indexes = None ):
        """ Function doc """
 
        if ind_vbo:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ind_vbo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indexes.nbytes, indexes, GL.GL_DYNAMIC_DRAW)
 
    def _load_color_vbo(self, colors):
        """ This function assigns the colors to
            be drawn by the function  draw_representation"""
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.col_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.nbytes, colors, GL.GL_STATIC_DRAW)

    def draw_background_sel_representation(self):
        pass
    
    def draw_representation(self):
        """ Function doc """
        frame_coords, frame = self.vm_glcore._safe_frame_coords(self.vismol_object)
        self.vertices, self.colors, self.indexes = self.vismol_object.surface_trajectory[frame][self.surf_name]
        
        self._check_vao_and_vbos()
        GL.glEnable(GL.GL_DEPTH_TEST)
        #GL.glEnable(GL.GL_CULL_FACE)
        #GL.glCullFace(GL.GL_BACK)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_BLEND)
        GL.glUseProgram(self.shader_program)
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_lights(self.shader_program)
        self.vm_glcore.load_fog(self.shader_program)
        GL.glBindVertexArray(self.vao)
        
        #self.vismol_object
        

        self._load_coord_vbo( coord_vbo=True, vertices = self.vertices)
        self._load_ind_vbo  ( ind_vbo = True, indexes = self.indexes)
        self._load_color_vbo( self.colors)
        
        #if self.was_rep_coord_modified or self.was_rep_ind_modified:
        #    coords, colors, rads = self._coords_colors_rads()
        #    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.insta_vbo)
        #    GL.glBufferData(GL.GL_ARRAY_BUFFER, coords.nbytes, coords, GL.GL_STATIC_DRAW)
        #    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.col_vbo)
        #    GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.nbytes, colors, GL.GL_STATIC_DRAW)
        #    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.rad_vbo)
        #    GL.glBufferData(GL.GL_ARRAY_BUFFER, rads.nbytes, rads, GL.GL_STATIC_DRAW)
        #    self.elements = np.uint32(coords.shape[0])
        #    self.was_rep_coord_modified = False
        #    self.was_rep_ind_modified = False
        
        #GL.glDrawElementsInstanced(GL.GL_TRIANGLES, self.instances_elemns, GL.GL_UNSIGNED_INT, None, self.elements)
        GL.glDrawElements(GL.GL_LINES, int(len(self.indexes)), GL.GL_UNSIGNED_INT, None)
        #GL.glDrawElements(GL.GL_TRIANGLES, int(len(self.indexes)), GL.GL_UNSIGNED_INT, None)
        #GL.glDrawElements(GL.GL_TRIANGLE_STRIP, int(len(self.indexes)), GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_DEPTH_TEST)

    def _check_vao_and_vbos(self):
        #print(self.name)
        self.shader_program = self.vm_glcore.shader_programs[self.name]
        #self.sel_shader_program = self.vm_glcore.shader_programs[self.name + "_sel"]
        if self.vao is None:
            self._make_gl_representation_vao_and_vbos()
        #if self.sel_vao is None:
        #    self._make_gl_sel_representation_vao_and_vbos()

class CartoonRepresentation(Representation):
    def __init__ (self, name = 'cartoon', active = True, rep_type = 'mol', vismol_object = None, vismol_glcore = None, indexes = []):
        self.name               = name
        self.active             = active
        self.type               = rep_type

        self.vm_object          = vismol_object
        self.vm_glcore          = vismol_glcore
        
        # representation 	
        self.vao            = None
        self.ind_vbo        = None
        self.coord_vbo      = None
        self.norm_vbo       = None
        self.col_vbo        = None
        self.size_vbo       = None
           

        # bgrd selection   
        self.sel_vao        = None
        self.sel_ind_vbo    = None
        self.sel_coord_vbo  = None
        self.sel_col_vbo    = None
        self.sel_size_vbo   = None


        #     S H A D E R S
        self.shader_program     = None
        self.sel_shader_program = None
        
        import vismol.utils.cartoon_BCK as cartoon
        
        coords, normals, indexes, colors = cartoon.cartoon(vismol_object, spline_detail=5)
        
        coords = coords.flatten()
        normals = normals.flatten()
        colors = colors.flatten()
        
        
        self.coords2 = coords
        self.colors2 = colors
        self.normals2 = normals
        self.indexes2 = indexes

    def _make_gl_vao_and_vbos (self, indexes = None):
        """ Function doc """
        #if indexes is not None:
        #    pass
        #else:
        
        #dot_qtty  = int(len(self.vm_object.frames[0])/3)
        #indexes = []
        #for i in range(dot_qtty):
        #    indexes.append(i)
        

        self.shader_program     = self.vm_glcore.shader_programs[self.name]
        #self.sel_shader_program = self.vm_glcore.shader_programs[self.name+'_sel']
        

        """
        coords  = np.array(self.coords2, dtype=np.float32)
        colors  = np.array(self.colors2, dtype=np.float32)
        normals = np.array(self.normals2, dtype=np.float32)
        indexes = np.array(self.indexes2, dtype=np.uint32)
        """
        
        
        coords  = self.coords2 
        colors  = self.colors2 
        normals = self.normals2
        indexes = self.indexes2
        
        print ('len(coords),len(colors), len(normals),len(indexes)', len(coords),len(colors), len(normals),len(indexes)  )

        self._make_gl_representation_vao_and_vbos (indexes    = indexes,
                                                   coords     = coords ,
                                                   colors     = colors ,
                                                   dot_sizes  = None   ,
                                                   normals    = normals
                                                   )
        
        
        
        self.ind_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ind_vbo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indexes.itemsize*len(indexes), indexes, GL.GL_DYNAMIC_DRAW)
        
        #self.coord_vbo = GL.glGenBuffers(1)
        #GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.coord_vbo)
        ##GL.glBufferData(GL.GL_ARRAY_BUFFER, coords.itemsize*len(coords), coords, GL.GL_STATIC_DRAW)
        #GL.glBufferData(GL.GL_ARRAY_BUFFER, coords.nbytes, coords, GL.GL_STATIC_DRAW)
        #gl_coord = GL.glGetAttribLocation(self.shader_program, 'vert_coord')
        #GL.glEnableVertexAttribArray(gl_coord)
        #GL.glVertexAttribPointer(gl_coord, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*coords.itemsize, ctypes.c_void_p(0))
        
        
        self.col_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.col_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.itemsize*len(colors), colors, GL.GL_STATIC_DRAW)
        gl_color = GL.glGetAttribLocation(self.shader_program, 'vert_color')
        GL.glEnableVertexAttribArray(gl_color)
        GL.glVertexAttribPointer(gl_color, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*colors.itemsize, ctypes.c_void_p(0))

        self.norm_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.norm_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, normals.itemsize*len(normals), normals, GL.GL_STATIC_DRAW)
        gl_norm = GL.glGetAttribLocation(self.shader_program, 'vert_norm')
        GL.glEnableVertexAttribArray(gl_norm)
        GL.glVertexAttribPointer(gl_norm, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*normals.itemsize, ctypes.c_void_p(0))
        
        
        
        
        
        #self.centr_vbo = GL.glGenBuffers(1)
        #GL.glBindBuffer(GL.GL_ARRAY_BUFFER, coords)
        #GL.glBufferData(GL.GL_ARRAY_BUFFER, coords.itemsize*len(coords), coords, GL.GL_STATIC_DRAW)
        #gl_center = GL.glGetAttribLocation(self.shader_program , 'vert_centr')
        #GL.glEnableVertexAttribArray(gl_center)
        #GL.glVertexAttribPointer(gl_center, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*coords.itemsize, ctypes.c_void_p(0))
        
        
        
        colors_idx = self.vm_object.color_indexes
        self.sel_vao = True
        """
        self._make_gl_sel_representation_vao_and_vbos (indexes    = indexes    ,
                                                       coords     = coords     ,
                                                       colors     = colors_idx ,
                                                       dot_sizes  = None       ,
                                                       )
        """

    def draw_representation (self):
        """ Function doc """
        self._check_vao_and_vbos ()
        #self._enable_anti_alias_to_lines()
        
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        #GL.glCullFace(GL.GL_BACK)
        view = self.vm_glcore.glcamera.view_matrix
        
        GL.glUseProgram(self.shader_program )
        
        #print (self.vm_object.model_mat,view)
        
        m_normal = np.array(np.matrix(np.dot(view, self.vm_object.model_mat)).I.T)
        
        self.vm_glcore.load_matrices(self.shader_program , self.vm_object.model_mat)
        self.vm_glcore.load_lights  (self.shader_program )
        self.vm_glcore.load_fog     (self.shader_program )
        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, int(len(self.indexes2)), GL.GL_UNSIGNED_INT, None)
        GL.glDisable(GL.GL_DEPTH_TEST)
        
        
        
        
        
        
        
        #"""
        ##print ("DotsRepresentation")
        #height = self.vm_glcore.height
        #
        #GL.glUseProgram(self.shader_program)
        ##1*self.height dot_size
        ##GL.glLineWidth(40/abs(self.vm_glcore.dist_cam_zrp))
        #GL.glPointSize(0.1*height/abs(self.vm_glcore.dist_cam_zrp)) # dot size not included yet
        #self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        #self.vm_glcore.load_fog(self.shader_program)
        #GL.glBindVertexArray(self.vao)
        #"""
        #if self.vm_glcore.modified_view:
        #    pass
        #
        #else:
        #    """
        #    This function checks if the number of the called frame will not exceed 
        #    the limit of frames that each object has. Allowing two objects with 
        #    different trajectory sizes to be manipulated at the same time within the 
        #    glArea"""
        #    # self._set_coordinates_to_buffer(coord_vbo = True, sel_coord_vbo = False)
        #    #GL.glDrawElements(GL.GL_POINTS, int(len(self.indexes2)), GL.GL_UNSIGNED_INT, None)
        #    #GL.glDrawElements(GL.GL_LINE_LOOP, int(len(self.coords2)), GL.GL_UNSIGNED_INT, None)
        #    #GL.glDrawElements(GL.GL_LINE_STRIP, int(len(self.indexes2)), GL.GL_UNSIGNED_INT, None)
        #    
        #    #print("int(len(self.indexes2))", int(len(self.indexes2)))
        #    GL.glDrawElements(GL.GL_TRIANGLES, int(len(self.indexes2)), GL.GL_UNSIGNED_INT, None)
        #    #GL.glDrawElements(GL.GL_TRIANGLES, 54060, GL.GL_UNSIGNED_INT, None)
        #
        ##GL.glBindVertexArray(0)
        ##GL.glLineWidth(1)
        ##GL.glUseProgram(0)
        ##GL.glDisable(GL.GL_LINE_SMOOTH)
        ##GL.glDisable(GL.GL_BLEND)
        #GL.glDisable(GL.GL_DEPTH_TEST)
                    
    def draw_background_sel_representation  (self):
        """ Function doc """
        pass







'''
class DynamicBonds(Representation):
    """ Class doc """
    
    def __init__ (self, vismol_object, vismol_glcore, name="sticks", indexes=None, active=True):
        """ Class initialiser """
        super(DynamicBonds, self).__init__(vismol_object, vismol_glcore, name, active, indexes)
    
    def draw_representation(self):
        """ Function doc """
        self._check_vao_and_vbos ()
        self._enable_anti_alias_to_lines()
        GL.glUseProgram(self.shader_program)
        GL.glLineWidth(40/abs(self.vm_glcore.dist_cam_zrp))
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_fog(self.shader_program)
        self.vm_glcore.load_lights(self.shader_program)
        GL.glBindVertexArray(self.vao)
        
        if self.vm_glcore.modified_view:
            pass
        else:
            # This function checks if the number of the called frame will not exceed 
            # the limit of frames that each object has. Allowing two objects with 
            # different trajectory sizes to be manipulated at the same time within the 
            # glArea
            frame = self.vm_glcore.frame
            if frame < len(self.vm_object.dynamic_bons):
                self.define_new_indexes_to_vbo(self.vm_object.dynamic_bons[frame])
                self._set_coordinates_to_buffer(coord_vbo=True, sel_coord_vbo=False)
                GL.glDrawElements(GL.GL_LINES, int(len(self.vm_object.dynamic_bons[frame])*2), GL.GL_UNSIGNED_INT, None)
            else:
                self.define_new_indexes_to_vbo(self.vm_object.dynamic_bons[-1])
                self._set_coordinates_to_buffer(coord_vbo=True, sel_coord_vbo=False)
                GL.glDrawElements(GL.GL_LINES, int(len(self.vm_object.dynamic_bons[-1])*2), GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        self._disable_anti_alias_to_lines()
        GL.glLineWidth(1)
        GL.glUseProgram(0)
    
    def draw_background_sel_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        self._disable_anti_alias_to_lines()
        GL.glUseProgram(self.sel_shader_program)
        GL.glLineWidth(20)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.vm_glcore.load_matrices(self.sel_shader_program, self.vm_object.model_mat)
        GL.glBindVertexArray(self.sel_vao)
        
        if self.vm_glcore.modified_view:
            pass
        else:
            # This function checks if the number of the called frame will not exceed 
            # the limit of frames that each object has. Allowing two objects with 
            # different trajectory sizes to be manipulated at the same time within the 
            # glArea
            frame = self.vm_glcore.frame
            if frame < len(self.vm_object.dynamic_bons):
                self.define_new_indexes_to_vbo(self.vm_object.dynamic_bons[frame])
                self._set_coordinates_to_buffer(coord_vbo=True, sel_coord_vbo=False)
                GL.glDrawElements(GL.GL_LINES, int(len(self.vm_object.dynamic_bons[frame])*2), GL.GL_UNSIGNED_INT, None)
            else:
                self.define_new_indexes_to_vbo(self.vm_object.dynamic_bons[-1])
                self._set_coordinates_to_buffer(coord_vbo=True, sel_coord_vbo=False)
                GL.glDrawElements(GL.GL_LINES, int(len(self.vm_object.dynamic_bons[-1])*2), GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(1)
        GL.glUseProgram(0)



class RibbonsRepresentation(Representation):
    """ Class doc """
    
    def __init__(self, vismol_object, vismol_glcore, name="ribbon", indexes=None, active=True):
        """ Class initialiser """
        super(RibbonsRepresentation, self).__init__(vismol_object, vismol_glcore, name, active, indexes)
        
        if self.vm_object.c_alpha_bonds == []:
            self.vm_object.get_backbone_indexes()
        
        indexes = []
        for bond in self.vm_object.c_alpha_bonds:
            indexes.append(bond.atom_index_i)
            indexes.append(bond.atom_index_j)
        
        if indexes == []:
            self.activate = False
        else:
            self.indexes = np.array(indexes, dtype=np.uint32)
    
    def draw_representation(self):
        """ Function doc """
        self._check_vao_and_vbos ()
        self._enable_anti_alias_to_lines()
        GL.glUseProgram(self.shader_program)
        ribbon_width = self.vm_session.vm_config.gl_parameters["ribbon_width"]
        ribbon_width = (ribbon_width*20)/abs(self.vm_glcore.dist_cam_zrp)/2
        GL.glLineWidth(ribbon_width)
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_fog(self.shader_program)
        GL.glBindVertexArray(self.vao)
        
        if self.vm_glcore.modified_view:
            pass
        else:
            # This function checks if the number of the called frame will not exceed 
            # the limit of frames that each object has. Allowing two objects with 
            # different trajectory sizes to be manipulated at the same time within the 
            # glArea
            self._set_coordinates_to_buffer(coord_vbo=True, sel_coord_vbo=False)
            GL.glDrawElements(GL.GL_LINES, int(len(self.vm_object.index_bonds)*2), GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        self._disable_anti_alias_to_lines()
        GL.glLineWidth(1)
        GL.glUseProgram(0)
    
    def draw_background_sel_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        self._disable_anti_alias_to_lines()
        line_width = self.vm_session.vm_config.gl_parameters["line_width_selection"] 
        GL.glUseProgram(self.sel_shader_program)
        GL.glLineWidth(line_width)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.vm_glcore.load_matrices(self.sel_shader_program, self.vm_object.model_mat)
        GL.glBindVertexArray(self.sel_vao)
        
        if self.vm_glcore.modified_view:
            pass
        else:
            # This function checks if the number of the called frame will not exceed 
            # the limit of frames that each object has. Allowing two objects with 
            # different trajectory sizes to be manipulated at the same time within the 
            # glArea
            self._set_coordinates_to_buffer(coord_vbo=False, sel_coord_vbo=True)
            GL.glDrawElements(GL.GL_LINES, int(len(self.vm_object.index_bonds)*2), GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(1)
        GL.glUseProgram(0)



class ImpostorRepresentation(Representation):
    """ Class doc """
    
    def __init__ (self, vismol_object, vismol_glcore, name = "impostor", indexes=None, active=True, scale=1.0):
        """ Class initialiser """
        super(ImpostorRepresentation, self).__init__(vismol_object, vismol_glcore, name, active, indexes)
        self.scale = scale
    
    def draw_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        self._enable_anti_alias_to_lines()
        GL.glUseProgram(self.shader_program)
        height = self.vm_glcore.height
        dist_cam_zrp = self.vm_glcore.dist_cam_zrp
        xyz_coords = self.vm_glcore.glcamera.get_modelview_position(self.vm_object.model_mat)
        u_campos = GL.glGetUniformLocation(self.shader_program, "u_campos")
        GL.glUniform3fv(u_campos, 1, xyz_coords)
        self.vm_glcore.load_lights(self.shader_program)
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_fog(self.shader_program)
        GL.glBindVertexArray(self.vao)

        if self.vm_glcore.modified_view:
            pass
        else:
            # This function checks if the number of the called frame will not exceed 
            # the limit of frames that each object has. Allowing two objects with 
            # different trajectory sizes to be manipulated at the same time within the 
            # glArea
            self._set_coordinates_to_buffer(coord_vbo=True, sel_coord_vbo=False)
            GL.glDrawElements(GL.GL_POINTS, len(self.vm_object.atoms), GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        self._disable_anti_alias_to_lines()
        GL.glPointSize(1)
        GL.glUseProgram(0)
    
    def draw_background_sel_representation(self):
        """ Function doc """
        self._check_vao_and_vbos()
        self._disable_anti_alias_to_lines()
        GL.glUseProgram(self.sel_shader_program)
        self.vm_glcore.load_matrices(self.sel_shader_program, self.vm_object.model_mat)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glBindVertexArray(self.sel_vao)
        
        if self.vm_glcore.modified_view:
            pass
        else:
            # This function checks if the number of the called frame will not exceed 
            # the limit of frames that each object has. Allowing two objects with 
            # different trajectory sizes to be manipulated at the same time within the 
            # glArea
            self._set_coordinates_to_buffer(coord_vbo=False, sel_coord_vbo=True)
            GL.glDrawElements(GL.GL_POINTS, len(self.vm_object.atoms), GL.GL_UNSIGNED_INT, None)
        
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glPointSize(1)
        GL.glUseProgram(0)
'''


'''
class SurfaceRepresentation(Representation):
    """ Class doc """
    
    def __init__ (self, name = "surface", active = True, rep_type = "mol", vismol_object = None, vm_glcore = None, indexes = []):
        """ Class initialiser """
        self.name               = name
        self.active             = active
        self.type               = rep_type

        self.vm_object             = vismol_object
        self.vm_glcore             = vm_glcore
        
        

        
        
        # representation 	
        self.vao            = None
        self.ind_vbo        = None
        self.coord_vbo      = None
        self.norm_vbo       = None
        self.col_vbo        = None
        self.size_vbo       = None
           

        # bgrd selection   
        self.sel_vao        = None
        self.sel_ind_vbo    = None
        self.sel_coord_vbo  = None
        self.sel_col_vbo    = None
        self.sel_size_vbo   = None


        #     S H A D E R S
        self.shader_program     = None
        self.sel_shader_program = None
        self.read_surface_data()
    
    
    ##### sub 2 vev3 vectors
    def sub_vec3(self, a, b):
        c = [ a[0] - b[0],
              a[1] - b[1],
              a[2] - b[2] ]

        return c

    ## add 2 vectors and take the avg
    ## if a vector is still 0 we just take b
    def avg_add_vec3(self, a, b):
        if a[0] == 0.0 and a[1] == 0.0 and a[2] == 0.0 :
            return b

        c = [ (a[0] + b[0]) * 0.5 ,
              (a[1] + b[1]) * 0.5 ,
              (a[2] + b[2]) * 0.5 ]

        return c    

    ## make the cross product of 2 vectors
    def cross_vec3(self, a, b):
        c = [a[1]*b[2] - a[2]*b[1],
             a[2]*b[0] - a[0]*b[2],
             a[0]*b[1] - a[1]*b[0]]

        return c
    #############################################
        
    
    
    def read_surface_data(self):
        """ Function doc """
        #from random import random 
        #
        #[verts, tris, verts_gpu, tris_gpu] = edtsurf.calc_surface("/home/fernando/programs/EasyHybrid3/Coords/pdbs/1bx4_H.pdb")
        #self.coords2  = verts_gpu
        #self.indexes2 = tris_gpu
        #self.colors2  = []
        #
        #
        #size = len( self.coords2 )
        #for i in range(size):
        #    self.colors2.append(float(i/size) + random())
        
        rawdata = open("../EasyHybrid3/Coords/pdbs/1bx4.ply", "r")
        lines  = rawdata.readlines()
        
        self.coords2 = []
        self.colors2 = []
        self.normals2 = []
        self.indexes2 = []
        avg_normals_indexes = []
        
        
        for line in lines:
            line2 = line.split()
            
            if len(line2) == 6:
                #print (line2)
                self.coords2.append(float(line2[0]))
                self.coords2.append(float(line2[1]))
                self.coords2.append(float(line2[2]))
                                                  
                self.colors2.append(float(line2[3])/255)
                self.colors2.append(float(line2[4])/255)
                self.colors2.append(float(line2[5])/255)
                
                self.normals2.append(float(line2[0]))
                self.normals2.append(float(line2[1]))
                self.normals2.append(float(line2[2]))                
                avg_normals_indexes.append( ( 0.0 , 0.0 , 0.0 ) )  ### NEW !!! 

            if len(line2) == 7:
                
                self.indexes2.append(int(line2[1]))
                self.indexes2.append(int(line2[2]))
                self.indexes2.append(int(line2[3]))
                
        
        ## calculate normals and interpolate them (thanks a lot Kai)
        for i in range( 0 , len(self.indexes2) , 3 ):

            index_1 = self.indexes2[i] * 3;
            index_2 = self.indexes2[i+1] * 3;
            index_3 = self.indexes2[i+2] * 3;
            vertex_1 = ( self.coords2[index_1] , self.coords2[index_1+1] , self.coords2[index_1+2] )
            vertex_2 = ( self.coords2[index_2] , self.coords2[index_2+1] , self.coords2[index_2+2] )
            vertex_3 = ( self.coords2[index_3] , self.coords2[index_3+1] , self.coords2[index_3+2] )

            vec_p0_p1 = self.sub_vec3( vertex_2 , vertex_1 )
            vec_p0_p2 = self.sub_vec3( vertex_3 , vertex_1 )
            norm_vec  = self.cross_vec3( vec_p0_p1, vec_p0_p2 )

            vert_index_1 = self.indexes2[i] ;
            vert_index_2 = self.indexes2[i+1] ;
            vert_index_3 = self.indexes2[i+2] ;
            
            avg_normals_indexes[vert_index_1] = self.avg_add_vec3( avg_normals_indexes[vert_index_1] , norm_vec )
            avg_normals_indexes[vert_index_2] = self.avg_add_vec3( avg_normals_indexes[vert_index_2] , norm_vec )
            avg_normals_indexes[vert_index_3] = self.avg_add_vec3( avg_normals_indexes[vert_index_3] , norm_vec )


        ## set all new interpolated normals   
        for i in range( 0 , len(self.indexes2) , 1 ):
            index_1 = self.indexes2[i] * 3;

            self.normals2[index_1]   = avg_normals_indexes[self.indexes2[i]][0]
            self.normals2[index_1+1] = avg_normals_indexes[self.indexes2[i]][1]
            self.normals2[index_1+2] = avg_normals_indexes[self.indexes2[i]][2]





               
                

    def _make_gl_vao_and_vbos (self, indexes = None):
        """ Function doc """
        #if indexes is not None:
        #    pass
        #else:
        
        #dot_qtty  = int(len(self.vm_object.frames[0])/3)
        #indexes = []
        #for i in range(dot_qtty):
        #    indexes.append(i)
        

        self.shader_program     = self.vm_glcore.shader_programs[self.name]
        self.sel_shader_program = self.vm_glcore.shader_programs[self.name+"_sel"]
        
        #indexes = np.array(self.vm_object.index_bonds, dtype=np.uint32)
        #indexes = np.array(self.vm_object.idex, dtype=np.uint32)

        coords  = np.array(self.coords2, dtype=np.float32)
        colors  = np.array(self.colors2, dtype=np.float32)
        normals = np.array(self.normals2, dtype=np.float32)
        #indexes = range(0, len(self.coords2))     
        #indexes = np.array(indexes, dtype=np.uint32)
        indexes = np.array(self.indexes2, dtype=np.uint32)
        #print (indexes)


        self._make_gl_representation_vao_and_vbos (indexes    = indexes,
                                                   coords     = coords ,
                                                   colors     = colors ,
                                                   dot_sizes  = None   ,
                                                   normals    = normals
                                                   )
        
        #self.centr_vbo = GL.glGenBuffers(1)
        #GL.glBindBuffer(GL.GL_ARRAY_BUFFER, coords)
        #GL.glBufferData(GL.GL_ARRAY_BUFFER, coords.itemsize*len(coords), coords, GL.GL_STATIC_DRAW)
        #gl_center = GL.glGetAttribLocation(self.shader_program , "vert_centr")
        #GL.glEnableVertexAttribArray(gl_center)
        #GL.glVertexAttribPointer(gl_center, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*coords.itemsize, ctypes.c_void_p(0))
        
        
        
        colors_idx = self.vm_object.color_indexes
        self._make_gl_sel_representation_vao_and_vbos (indexes    = indexes    ,
                                                       coords     = coords     ,
                                                       colors     = colors_idx ,
                                                       dot_sizes  = None       ,
                                                       )

    def draw_representation (self):
        """ Function doc """
        self._check_vao_and_vbos ()
        #self._enable_anti_alias_to_lines()
        
        
        
        
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        view = self.vm_glcore.glcamera.view_matrix
        
        GL.glUseProgram(self.shader_program )
        
        #print (self.vm_object.model_mat,view)
        
        m_normal = np.array(np.matrix(np.dot(view, self.vm_object.model_mat)).I.T)
        
        self.vm_glcore.load_matrices(self.shader_program , self.vm_object.model_mat)
        self.vm_glcore.load_lights  (self.shader_program )
        self.vm_glcore.load_fog     (self.shader_program )
        GL.glBindVertexArray(self.vao)
        
        
        
        
        
        
        
        
        
        """
        #print ("DotsRepresentation")
        height = self.vm_glcore.height
        
        GL.glUseProgram(self.shader_program)
        #1*self.height dot_size
        #GL.glLineWidth(40/abs(self.vm_glcore.dist_cam_zrp))
        GL.glPointSize(0.1*height/abs(self.vm_glcore.dist_cam_zrp)) # dot size not included yet
        self.vm_glcore.load_matrices(self.shader_program, self.vm_object.model_mat)
        self.vm_glcore.load_fog(self.shader_program)
        GL.glBindVertexArray(self.vao)
        """
        if self.vm_glcore.modified_view:
            pass
        
        else:
            """
            This function checks if the number of the called frame will not exceed 
            the limit of frames that each object has. Allowing two objects with 
            different trajectory sizes to be manipulated at the same time within the 
            glArea"""
            # self._set_coordinates_to_buffer(coord_vbo = True, sel_coord_vbo = False)
            #GL.glDrawElements(GL.GL_POINTS, int(len(self.indexes2)), GL.GL_UNSIGNED_INT, None)
            #GL.glDrawElements(GL.GL_LINE_LOOP, int(len(self.coords2)), GL.GL_UNSIGNED_INT, None)
            #GL.glDrawElements(GL.GL_LINE_STRIP, int(len(self.indexes2)), GL.GL_UNSIGNED_INT, None)
            GL.glDrawElements(GL.GL_TRIANGLES, int(len(self.indexes2)), GL.GL_UNSIGNED_INT, None)
        
        #GL.glBindVertexArray(0)
        #GL.glLineWidth(1)
        #GL.glUseProgram(0)
        #GL.glDisable(GL.GL_LINE_SMOOTH)
        #GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_DEPTH_TEST)
        
            
    def draw_background_sel_representation  (self):
        """ Function doc """
        pass


class WiresRepresentation(Representation):
    """ Class doc """
    
    def __init__ (self, name = "wires", active = True, rep_type = "mol", vismol_object = None, vm_glcore = None, indexes = []):
        """ Class initialiser """
        self.name               = name
        self.active             = active
        self.type               = rep_type
        self.vm_object             = vismol_object
        self.vm_glcore             = vm_glcore
        
        # representation    
        self.vao            = None
        self.ind_vbo        = None
        self.coord_vbo      = None
        self.col_vbo        = None
        self.size_vbo       = None
        
        # bgrd selection   
        self.sel_vao        = None
        self.sel_ind_vbo    = None
        self.sel_coord_vbo  = None
        self.sel_col_vbo    = None
        self.sel_size_vbo   = None

        #     S H A D E R S
        self.shader_program     = None
        self.sel_shader_program = None
        self.read_surface_data()
    
    def read_surface_data(self):
        """ Function doc """
        rawdata = open("../EasyHybrid3/Coords/pdbs/1bx4.ply", "r")
        lines  = rawdata.readlines()
        
        self.coords2 = []
        self.colors2 = []
        self.indexes2 = []
        
        for line in lines:
            line2 = line.split()
            if len(line2) == 6:
                self.coords2.append(float(line2[0]))
                self.coords2.append(float(line2[1]))
                self.coords2.append(float(line2[2]))
                self.colors2.append(float(line2[3])/255)
                self.colors2.append(float(line2[4])/255)
                self.colors2.append(float(line2[5])/255)
            if len(line2) == 7:
                self.indexes2.append(int(line2[1]))
                self.indexes2.append(int(line2[2]))
                self.indexes2.append(int(line2[3]))

    def _make_gl_vao_and_vbos (self, indexes = None):
        """ Function doc """
        self.shader_program     = self.vm_glcore.shader_programs[self.name]
        self.sel_shader_program = self.vm_glcore.shader_programs[self.name+"_sel"]
        coords  = np.array(self.coords2, dtype=np.float32)
        colors  = np.zeros(len(self.colors2))
        indexes = np.array(self.indexes2, dtype=np.uint32)
        self._make_gl_representation_vao_and_vbos (indexes    = indexes,
                                                   coords     = coords ,
                                                   colors     = colors ,
                                                   dot_sizes  = None   ,
                                                   )
        colors_idx = self.vm_object.color_indexes
        self._make_gl_sel_representation_vao_and_vbos (indexes    = indexes    ,
                                                       coords     = coords     ,
                                                       colors     = colors_idx ,
                                                       dot_sizes  = None       ,
                                                       )

    def draw_representation (self):
        """ Function doc """
        self._check_vao_and_vbos ()
        pass
        #GL.glEnable(GL.GL_DEPTH_TEST)
        #GL.glEnable(GL.GL_CULL_FACE)
        #GL.glCullFace(GL.GL_BACK)
        #
        ##LineWidth = (80/abs(self.vm_glcore.dist_cam_zrp)/2)**0.5  #40/abs(self.vm_glcore.dist_cam_zrp)
        ##GL.glLineWidth(2)
        #
        #
        #view = self.vm_glcore.glcamera.view_matrix
        #GL.glUseProgram(self.shader_program )
        #m_normal = np.array(np.matrix(np.dot(view, self.vm_object.model_mat)).I.T)
        #self.vm_glcore.load_matrices(self.shader_program , self.vm_object.model_mat)
        ##self.vm_glcore.load_lights  (self.shader_program )
        #self.vm_glcore.load_fog     (self.shader_program )
        #GL.glBindVertexArray(self.vao)
        #if self.vm_glcore.modified_view:
        #    pass
        #
        #else:
        #    """
        #    This function checks if the number of the called frame will not exceed 
        #    the limit of frames that each object has. Allowing two objects with 
        #    different trajectory sizes to be manipulated at the same time within the 
        #    glArea"""
        #    # self._set_coordinates_to_buffer(coord_vbo = True, sel_coord_vbo = False)
        #    GL.glDrawElements(GL.GL_TRIANGLES, int(len(self.indexes2)), GL.GL_UNSIGNED_INT, None)
        #GL.glDisable(GL.GL_DEPTH_TEST)
        
    def draw_background_sel_representation  (self):
        """ Function doc """
        pass


class LabelRepresentation:
    """ Class doc """
    
    def __init__ (self, name = "labels", active = True, rep_type = "mol", vismol_object = None, vm_glcore = None, indexes = []):
        """ Class initialiser """
        self.vm_object = vismol_object
        self.name   = name
        self.active = True
        self.vm_glcore = vm_glcore
        
        self.chars     = 0 
        #self._check_vao_and_vbos()
        
    def _check_vao_and_vbos (self, indexes = None):
        """ Function doc """
        if self.vm_object.vm_font.vao is None:
            self.vm_object.vm_font.make_freetype_font()
            self.vm_object.vm_font.make_freetype_texture(self.vm_glcore.freetype_program)
        
        if self.chars == 0:
            print("self._build_buffer()")
            self._build_buffer()
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vm_object.vm_font.vbos[0])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.xyz_pos.itemsize*len(self.xyz_pos), self.xyz_pos, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vm_object.vm_font.vbos[1])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.uv_coords.itemsize*len(self.uv_coords), self.uv_coords, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)


    def _build_buffer (self, indexes = None):
        self.chars     = 0
        self.xyz_pos   = []
        self.uv_coords = []
        for atom in self.vm_object.atoms:
            
            texto = atom.name
            point = np.array(atom.coords (self.vm_glcore.frame),np.float32)
            point = np.array((point[0],point[1],point[2],1),np.float32)
            point = np.dot(point, self.vm_object.model_mat)

            GL.glBindTexture(GL.GL_TEXTURE_2D, self.vm_object.vm_font.texture_id)
            for i,c in enumerate(texto):
                self.chars += 1
                c_id = ord(c)
                x = c_id%16
                y = c_id//16-2
                self.xyz_pos.append(point[0]+i*self.vm_object.vm_font.char_width)
                self.xyz_pos.append(point[1])
                self.xyz_pos.append(point[2])

                self.uv_coords.append(x*self.vm_object.vm_font.text_u)
                self.uv_coords.append(y*self.vm_object.vm_font.text_v)
                self.uv_coords.append((x+1)*self.vm_object.vm_font.text_u)
                self.uv_coords.append((y+1)*self.vm_object.vm_font.text_v)
            #print(texto)
        #print("xyz_pos  ",len(self.xyz_pos))
        #print("uv_coords",len(self.uv_coords))
        #print("atoms    ",len(self.vm_object.atoms))
        #print("chars    ",self.chars)
        
        self.xyz_pos   = np.array(self.xyz_pos  , np.float32)
        self.uv_coords = np.array(self.uv_coords, np.float32)
        


    
    
    def draw_representation (self):
        """ Function doc """
        self._check_vao_and_vbos()
        
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glUseProgram(self.vm_glcore.freetype_program)
        
        self.vm_object.vm_font.load_matrices(self.vm_glcore.freetype_program, self.vm_glcore.glcamera.view_matrix, self.vm_glcore.glcamera.projection_matrix)
        self.vm_object.vm_font.load_font_params(self.vm_glcore.freetype_program)
        
        GL.glBindVertexArray(self.vm_object.vm_font.vao)
        GL.glDrawArrays(GL.GL_POINTS, 0, self.chars)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

        

    def draw_background_sel_representation  (self):
        """ Function doc """
        pass

'''
