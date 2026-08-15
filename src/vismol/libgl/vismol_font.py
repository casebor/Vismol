#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  vismol_font.py
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

from vismol.utils.debug import dprint
import numpy as np
import freetype as ft
import ctypes
from OpenGL import GL
import os
import vismol.libgl.glaxis as glaxis
_fontdir = os.path.split(glaxis.__file__)[:-1]
FONTS_DIR = os.path.join(*_fontdir, "fonts")
#fontpath = os.path.join(*fontpath, "fonts", "VeraMono.ttf")
fontpath = os.path.join(FONTS_DIR, "Amiko-SemiBold.ttf")
DEFAULT_FONT_FILE = "Amiko-SemiBold.ttf"
DEFAULT_FONT_SIZE = 0.35


def list_available_fonts():
    """ Returns a sorted list of the .ttf font filenames bundled with
        VisMol (found in the libgl/fonts folder), e.g.
        ['Amiko-Bold.ttf', 'Amiko-Regular.ttf', ...].
    """
    try:
        fonts = [f for f in os.listdir(FONTS_DIR) if f.lower().endswith(".ttf")]
    except OSError:
        fonts = [DEFAULT_FONT_FILE]
    return sorted(fonts)


def resolve_font_path(font_name_or_path):
    """ Resolves a font "name" (as stored in the preferences, e.g.
        'Amiko-SemiBold.ttf') to a full path. If a full/relative path is
        already given (and exists) it is returned unchanged. Falls back to
        the default bundled font when the requested one can't be found.
    """
    if not font_name_or_path:
        return fontpath
    if os.path.isabs(font_name_or_path) and os.path.isfile(font_name_or_path):
        return font_name_or_path
    candidate = os.path.join(FONTS_DIR, os.path.basename(font_name_or_path))
    if os.path.isfile(candidate):
        return candidate
    return fontpath


class VismolFont():
    """ VismolFont stores the data created using the freetype python binding
        library, such as filename, character width, character height, character
        resolution, font color, etc.
    """
    
    def __init__(self, vismol_object=None, font_file=fontpath, char_res=64,
                 char_width=0.35, char_height=0.35, color=None):
                            
        """ Class initialiser
        """
        if color is None:
            color = [1, 1, 1, 1]
        self.vm_object = vismol_object
        self.font_file = font_file
        self.char_res = char_res
        self.char_width = char_width
        self.char_height = char_height
        self.offset = np.array([char_width/1.5, char_height/1.5], dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.font_buffer = None
        self.texture_id = None
        self.text_u = None
        self.text_v = None
        self.vao = None
        self.text_vbo = None
        self.coord_vbo = None
    
    def make_freetype_font(self):
        """ Function doc
        """
        face = ft.Face(self.font_file)
        face.set_char_size(self.char_res*64)
        # Determine largest glyph size
        width, height, ascender, descender = 0, 0, 0, 0
        for c in range(32,128):
            face.load_char(chr(c), ft.FT_LOAD_RENDER | ft.FT_LOAD_FORCE_AUTOHINT)
            bitmap = face.glyph.bitmap
            width = max(width, bitmap.width)
            ascender = max(ascender, face.glyph.bitmap_top)
            descender = max(descender, bitmap.rows-face.glyph.bitmap_top)
        height = ascender+descender
        # Generate texture data
        #self.font_buffer = np.zeros((height*6, width*16), dtype=np.ubyte)
        self.font_buffer = np.zeros((height*6, width*16), dtype=np.ubyte)
        for j in range(6):
            for i in range(16):
                face.load_char(chr(32+j*16+i), ft.FT_LOAD_RENDER | ft.FT_LOAD_FORCE_AUTOHINT )
                bitmap = face.glyph.bitmap
                x = i*width  + face.glyph.bitmap_left
                y = j*height + ascender - face.glyph.bitmap_top
                self.font_buffer[y:y+bitmap.rows,x:x+bitmap.width].flat = bitmap.buffer
        # Bound texture
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RED, self.font_buffer.shape[1], self.font_buffer.shape[0], 0, GL.GL_RED, GL.GL_UNSIGNED_BYTE, self.font_buffer)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        # Fill the font variables with data
        self.text_u = width/float(self.font_buffer.shape[1])
        self.text_v = height/float(self.font_buffer.shape[0])
    
    def make_freetype_texture(self, program):
        """ Function doc
        """
        coords = np.zeros(3, dtype=np.float32)
        uv_pos = np.zeros(4, dtype=np.float32)
        
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        
        coord_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, coord_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, coords.itemsize*len(coords), coords, GL.GL_DYNAMIC_DRAW)
        gl_coord = GL.glGetAttribLocation(program, "vert_coord")
        GL.glEnableVertexAttribArray(gl_coord)
        GL.glVertexAttribPointer(gl_coord, 3, GL.GL_FLOAT, GL.GL_FALSE, 3*coords.itemsize, ctypes.c_void_p(0))
        
        text_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, text_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, uv_pos.itemsize*len(uv_pos), uv_pos, GL.GL_DYNAMIC_DRAW)
        gl_texture = GL.glGetAttribLocation(program, "vert_uv")
        GL.glEnableVertexAttribArray(gl_texture)
        GL.glVertexAttribPointer(gl_texture, 4, GL.GL_FLOAT, GL.GL_FALSE, 4*uv_pos.itemsize, ctypes.c_void_p(0))
        
        GL.glBindVertexArray(0)
        GL.glDisableVertexAttribArray(gl_coord)
        GL.glDisableVertexAttribArray(gl_texture)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        
        self.vao = vao
        self.text_vbo = text_vbo
        self.coord_vbo = coord_vbo
    
    def _get_uniform_location(self, program, name):
        """ Cached wrapper around glGetUniformLocation. Uniform locations
            don't change once a program is linked, so we look each one up
            once per (program, name) pair instead of every frame.
        """
        if not hasattr(self, "_uniform_loc_cache"):
            self._uniform_loc_cache = {}
        key = (program, name)
        loc = self._uniform_loc_cache.get(key)
        if loc is None:
            loc = GL.glGetUniformLocation(program, name)
            self._uniform_loc_cache[key] = loc
        return loc
    
    def load_matrices(self, program, view_mat, proj_mat):
        """ Function doc """
        view = self._get_uniform_location(program, "view_mat")
        GL.glUniformMatrix4fv(view, 1, GL.GL_FALSE, view_mat)
        proj = self._get_uniform_location(program, "proj_mat")
        GL.glUniformMatrix4fv(proj, 1, GL.GL_FALSE, proj_mat)
    
    def load_font_params(self, program):
        """ Loads the uniform parameters for the OpenGL program, such as the
            offset coordinates (X,Y) to calculate the quad and the color of
            the font.
        """
        offset = self._get_uniform_location(program, "offset")
        GL.glUniform2fv(offset, 1, self.offset)
        color = self._get_uniform_location(program, "text_color")
        GL.glUniform4fv(color, 1, self.color)
        return True
    
    def print_all(self):
        """ Function created only with debuging purposes.
        """
        dprint("#############################################")
        dprint(self.font_file, "font_file")
        dprint(self.char_res, "char_res")
        dprint(self.char_width, "char_width")
        dprint(self.char_height, "char_height")
        dprint(self.offset, "offset")
        dprint(self.color, "color")
        dprint(self.font_buffer, "font_buffer")
        dprint(self.texture_id, "texture_id")
        dprint(self.text_u, "text_u")
        dprint(self.text_v, "text_v")
        dprint(self.vao, "vao")
        dprint(self.text_vbo, "text_vbo")
        dprint(self.coord_vbo, "coord_vbo")
    
    
    def set_dimensions (self, width, height ):
        """ Function doc """
        self.char_width  =  width
        self.char_height =  height
        self.offset = np.array([ width/2.0,  height/2.0], dtype=np.float32)
    
    def apply_settings(self, font_file=None, size=None):
        """ Updates the font file and/or the character size (width/height
            are kept equal, driven by a single "size" value) and marks the
            OpenGL texture/VAO for regeneration on the next draw call
            (vao=None). Used by the Preferences window to let the user
            change the font family and font size used for every label
            drawn in the glArea (atom labels, picking labels, distance
            labels).
        """
        if font_file is not None:
            self.font_file = resolve_font_path(font_file)
        if size is not None:
            self.set_dimensions(size, size)
        # Force the texture/VAO/font bitmap to be rebuilt on next use
        self.vao = None
        self.font_buffer = None
        self.texture_id = None
    
    def set_color (self, r = 1.0, g = 1.0, b = 1.0):      
        self.color = np.array([r,g,b], dtype=np.float32)
    
    
    def draw_labels(self):
        """ Function doc """
        pass
