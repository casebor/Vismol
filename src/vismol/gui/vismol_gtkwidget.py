#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  vismol_gtkwidget.py
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

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk
from gi.repository import GdkPixbuf

from logging import getLogger
from vismol.libgl.vismol_glcore import VismolGLCore
from vismol.gui.filechooser import FileChooser
logger = getLogger(__name__)

from OpenGL.GL import *
import numpy as np
from PIL import Image, ImageFilter, ImageOps

class VismolGTKWidget(Gtk.GLArea):
    """ Object that contains the GLArea from GTK3+.
        It needs a vertex and shader to be created, maybe later I"ll
        add a function to change the shaders.
    """
    
    def __init__(self, vismol_session=None, width=640.0, height=420.0):
        """ Class initialiser
        """
        super(VismolGTKWidget, self).__init__()
        self.connect("realize", self.initialize)
        self.connect("render", self.render)
        self.connect("resize", self.reshape)
        self.connect("key-press-event", self.key_pressed)
        self.connect("key-release-event", self.key_released)
        self.connect("button-press-event", self.mouse_pressed)
        self.connect("button-release-event", self.mouse_released)
        self.connect("motion-notify-event", self.mouse_motion)
        self.connect("scroll-event", self.mouse_scroll)
        self.grab_focus()
        self.set_events(self.get_events() | Gdk.EventMask.SCROLL_MASK
                        | Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK
                        | Gdk.EventMask.POINTER_MOTION_MASK | Gdk.EventMask.POINTER_MOTION_HINT_MASK
                        | Gdk.EventMask.KEY_PRESS_MASK | Gdk.EventMask.KEY_RELEASE_MASK)
        # self.vm_objects_list_store = Gtk.ListStore(bool,  # visible? 
        #                                            str,  # id
        #                                            str,  # name
        #                                            str,  # num of atoms
        #                                            str)  # num of frames
        self.vm_selection_modes_list_store = Gtk.ListStore(str)
        self.vm_session = vismol_session
        self.vm_glcore = VismolGLCore(self, vismol_session, width, height)
        self.glmenu_bg = None
        self.glmenu_sele = None
        self.glmenu_obj = None
        self.glmenu_pick = None
        self.filechooser = None
        self.selection_box_frame = None
    
    def initialize(self, widget):
        """ Enables the buffers and other charasteristics of the OpenGL context.
            sets the initial projection and view matrix
            
            self.flag -- Needed to only create one OpenGL program, otherwise a bunch of
                         programs will be created and use system resources. If the OpenGL
                         program will be changed change this value to True
        """
        if self.get_error() != None:
            print(self.get_error().args)
            print(self.get_error().code)
            print(self.get_error().domain)
            print(self.get_error().message)
            Gtk.main_quit()
        self.vm_glcore.initialize()
    
    def reshape(self, widget, width, height):
        """ Resizing function, takes the widht and height of the widget
            and modifies the view in the camera acording to the new values
        
            Keyword arguments:
            widget -- The widget that is performing resizing
            width -- Actual width of the window
            height -- Actual height of the window
        """
        self.vm_glcore.resize_window(width, height)
        self.queue_draw()
    
    def render(self, area, context):
        """ This is the function that will be called everytime the window
            needs to be re-drawed.
        """
        self.vm_glcore.render()
    
    def mouse_pressed(self, widget, event):
        """ Function doc """
        self.vm_glcore.mouse_pressed(event.button, event.x, event.y)
    
    def mouse_released(self, widget, event):
        """ Function doc """
        self.vm_glcore.mouse_released(event.button, event.x, event.y)
    
    def mouse_motion(self, widget, event):
        """ Function doc """
        self.vm_glcore.mouse_motion(event.x, event.y)
    
    def mouse_scroll(self, widget, event):
        """ Function doc
        """
        if event.direction == Gdk.ScrollDirection.UP:
            self.vm_glcore.mouse_scroll(1)
        if event.direction == Gdk.ScrollDirection.DOWN:
            self.vm_glcore.mouse_scroll(-1)
    
    def _build_glmenu(self, bg_menu=None, sele_menu=None, obj_menu=None, pick_menu=None):
        """ Function doc """
        if bg_menu is not None:
            self.glmenu_bg = Gtk.Menu()
            self.glmenu_bg_toplabel = Gtk.MenuItem(label="background")
            self._build_glmenu_from_dicts(bg_menu, self.glmenu_bg)
            self.glmenu_bg.show_all()
        else:
            self.glmenu_bg = None
        
        if sele_menu is not None:
            self.glmenu_sele = Gtk.Menu()
            self.glmenu_sele_toplabel = Gtk.MenuItem(label="selection")
            self._build_glmenu_from_dicts(sele_menu, self.glmenu_sele)
            self.glmenu_sele.show_all()
        else:
            self.glmenu_sele = None
        
        if pick_menu is not None:
            self.glmenu_pick = Gtk.Menu()
            self.glmenu_pick_toplabel = Gtk.MenuItem(label="picking")
            self.glmenu_pick.append(self.glmenu_pick_toplabel)
            self._build_glmenu_from_dicts(pick_menu, self.glmenu_pick)
            self.glmenu_pick.show_all()
        else:
            self.glmenu_pick = None
        
        if obj_menu is not None:
            self.glmenu_obj = Gtk.Menu()
            self.glmenu_obj_toplabel = Gtk.MenuItem(label="object")
            self.glmenu_obj.append(self.glmenu_obj_toplabel)
            self._build_glmenu_from_dicts(obj_menu, self.glmenu_obj)
            self.glmenu_obj.show_all()
        else:
            self.glmenu_obj = None
     
    def open_file(self, widget):
        """ Function doc """
        if self.filechooser is None:
            self.filechooser = FileChooser()
        filename = self.filechooser.open()
        self.vm_session.load_molecule(filename)
    
    def key_pressed(self, widget, event):
        """ The key_pressed function serves, as the names states, to catch
            events in the keyboard, e.g. letter "l" pressed, "backslash"
            pressed. Note that there is a difference between "A" and "a".
            Here I use a specific handler for each key pressed after
            discarding the CONTROL, ALT and SHIFT keys pressed (usefull
            for customized actions) and maintained, i.e. it"s the same as
            using Ctrl+Z to undo an action.
        """
        try:
            func = getattr(self, "_pressed_" + Gdk.keyval_name(event.keyval))
            func()
        except AttributeError as ae:
            logger.debug("Press key {} has not been assigned to a handler "\
                         "yet".format(Gdk.keyval_name(event.keyval)))
    
    def key_released(self, widget, event):
        """ Used to indicates a key has been released.
        """
        try:
            func = getattr(self, "_released_" + Gdk.keyval_name(event.keyval))
            func()
        except AttributeError as ae:
            logger.debug("Release key {} has not been assigned to a handler "\
                         "yet".format(Gdk.keyval_name(event.keyval)))
    
    def _pressed_Escape(self):
        """ Function doc """
        self.quit()
    
    def _pressed_Right(self):
        """ Function doc """
        self.vm_session.forward_frame()
        self.queue_draw()

    def _released_Right(self):
        """ Function doc """
        pass
    
    def _pressed_Left(self):
        """ Function doc """
        self.vm_session.reverse_frame()
        self.queue_draw()
    
    def _released_Left(self):
        """ Function doc """
        pass
    
    def _pressed_Control_L(self):
        """ Function doc """
        self.vm_glcore.ctrl = True
    
    def _released_Control_L(self):
        """ Function doc """
        self.vm_glcore.ctrl = False
    
    def _pressed_Shift_L(self):
        """ Function doc """
        self.vm_glcore.shift = True
    
    def _released_Shift_L(self):
        """ Function doc """
        self.vm_glcore.shift = False
    
    def _selection_type_picking(self, widget):
        if self.selection_box_frame:
            self.selection_box_frame.change_toggle_button_selecting_mode_status(True)
        else:
            self.vm_session.picking_selection_mode = True
        self.queue_draw()
    
    def _selection_type_viewing(self, widget):
        if self.selection_box_frame:
            self.selection_box_frame.change_toggle_button_selecting_mode_status(False)
        else:
            self.vm_session.picking_selection_mode = False
        self.queue_draw()
    
    def quit(self):
        logger.info("Thanks for using our software :). Quitting Vismol.")
        Gtk.main_quit()
    
    def _viewing_selection_mode_atom(self, widget):
        self.vm_session.viewing_selection_mode(sel_type="atom")
    
    def _viewing_selection_mode_residue(self, widget):
        self.vm_session.viewing_selection_mode(sel_type="residue")
    
    def _viewing_selection_mode_chain(self, widget):
        self.vm_session.viewing_selection_mode(sel_type="chain")
    
    def menu_show_dots(self, widget):
        self.vm_session.show_or_hide(rep_type="dots", show=True)
    
    def menu_hide_dots(self, widget):
        self.vm_session.show_or_hide(rep_type="dots", show=False)
    
    def menu_show_lines(self, widget):
        self.vm_session.show_or_hide(rep_type="lines", show=True)
    
    def menu_hide_lines(self, widget):
        self.vm_session.show_or_hide(rep_type="lines", show=False)
    
    def menu_show_nonbonded(self, widget):
        self.vm_session.show_or_hide(rep_type="nonbonded", show=True)
    
    def menu_hide_nonbonded(self, widget):
        self.vm_session.show_or_hide(rep_type="nonbonded", show=False)
    
    def menu_show_impostor(self, widget):
        self.vm_session.show_or_hide(rep_type="impostor", show=True)
    
    def menu_hide_impostor(self, widget):
        self.vm_session.show_or_hide(rep_type="impostor", show=False)
    
    def menu_show_spheres(self, widget):
        self.vm_session.show_or_hide(rep_type="spheres", show=True)
    
    def menu_hide_spheres(self, widget):
        self.vm_session.show_or_hide(rep_type="spheres", show=False)
    
    def menu_show_sticks(self, widget):
        self.vm_session.show_or_hide(rep_type="sticks", show=True)
    
    def menu_hide_sticks(self, widget):
        self.vm_session.show_or_hide(rep_type="sticks", show=False)
    
    def invert_selection(self, widget):
        self.vm_session.selections[self.vm_session.current_selection].invert_selection()
    
    def insert_glmenu(self, bg_menu=None, sele_menu=None, obj_menu=None, pick_menu=None):
        """ Function doc """
        if bg_menu is None:
            """ Standard Bg Menu"""
            bg_menu = {"Open File": ["MenuItem", self.open_file],
                       "separator": ["separator", None],
                       "Selection Mode": ["submenu",
                                            {"by atom": ["MenuItem", self._viewing_selection_mode_atom],
                                             "by residue": ["MenuItem", self._viewing_selection_mode_residue],
                                             "by chain": ["MenuItem", self._viewing_selection_mode_chain],
                                            }
                                         ],
                       "Selection Type": ["submenu",
                                            {"viewing": ["MenuItem", self._selection_type_viewing],
                                             "picking": ["MenuItem", self._selection_type_picking],
                                            }
                                         ],
                       "separator": ["separator", None],
                       "Quit": ["MenuItem", self.quit],
                      }
        
        if sele_menu is None:
            """ Standard Sele Menu """
            sele_menu = {"Show": ["submenu",
                                    {"dots": ["MenuItem", self.menu_show_dots],
                                     "lines": ["MenuItem", self.menu_show_lines],
                                     "nonbonded": ["MenuItem", self.menu_show_nonbonded],
                                     "sticks": ["MenuItem", self.menu_show_sticks],
                                     "impostor": ["MenuItem", self.menu_show_impostor],
                                     "spheres": ["MenuItem", self.menu_show_spheres],
                                    }
                                 ],
                         "Hide": ["submenu",
                                    {"dots": ["MenuItem", self.menu_hide_dots],
                                     "lines": ["MenuItem", self.menu_hide_lines],
                                     "nonbonded": ["MenuItem", self.menu_hide_nonbonded],
                                     "sticks": ["MenuItem", self.menu_hide_sticks],
                                     "impostor": ["MenuItem", self.menu_hide_impostor],
                                     "spheres": ["MenuItem", self.menu_hide_spheres],
                                    }
                                 ],
                         "separator":["separator", None],
                         "Invert Selection": ["MenuItem", self.invert_selection],
                        }
        
        if obj_menu is None:
            """ Standard Obj Menu"""
            obj_menu = {"Show": ["submenu",
                                    {"dots": ["MenuItem", None],
                                     "lines": ["MenuItem", None],
                                     "nonbonded": ["MenuItem", None],
                                    }
                                ],
                        "Hide": ["submenu",
                                    {"dots": ["MenuItem", None],
                                     "lines": ["MenuItem", None],
                                     "nonbonded": ["MenuItem", None],
                                    }
                                ],
                        "separator":["separator", None],
                        "label": ["submenu",
                                    {"Atom": ["submenu",
                                                {"index": ["MenuItem", None],
                                                 "name": ["MenuItem", None],
                                                 "residue": ["MenuItem", None],
                                                 "chain": ["MenuItem", None],
                                                 }
                                             ],
                                     "Residue": ["submenu",
                                                    {"index": ["MenuItem", None],
                                                     "name": ["MenuItem", None],
                                                     "chain": ["MenuItem", None],
                                                     }
                                                ],
                                     "Chain": ["submenu",
                                                {"name": ["MenuItem", None],
                                                 }
                                              ],
                                    },
                                 ],
                        }
        
        if pick_menu is None:
            """ Standard Sele Menu """
            pick_menu = {"Show": ["submenu",
                                    {"dots": ["MenuItem", None],
                                     "lines": ["MenuItem", None],
                                     "nonbonded": ["MenuItem", None],
                                    }
                                  ],
                         "Hide": ["submenu",
                                    {"dots": ["MenuItem", None],
                                     "lines": ["MenuItem", None],
                                     "nonbonded": ["MenuItem", None],
                                    }
                                  ],
                        }
        
        self._build_glmenu(bg_menu=bg_menu, sele_menu=sele_menu, obj_menu=obj_menu, pick_menu=pick_menu)
    
    def show_gl_menu(self, signals=None, menu_type=None, info=None):
        """ Function doc """
        if menu_type == "bg_menu":
            if self.glmenu_bg:
                self.glmenu_bg.popup(None, None, None, None, 0, 0)
        
        if menu_type == "sele_menu":
            if self.glmenu_sele:
                self.glmenu_sele.popup(None, None, None, None, 0, 0)
        
        if menu_type == "pick_menu":
            if self.glmenu_pick:
                self.glmenu_pick.popup(None, None, None, None, 0, 0)
        
        if menu_type == "obj_menu":
            if self.glmenu_obj:
                self.glmenu_obj_toplabel.set_label(info)
                self.glmenu_obj.popup(None, None, None, None, 0, 0)
    
    def _build_submenus_from_dicts(self, menu_dict):
        """ Function doc """
        menu = Gtk.Menu()
        for key in menu_dict:
            mitem = Gtk.MenuItem(key)
            if menu_dict[key][0] == "submenu":
                menu2 = self._build_submenus_from_dicts(menu_dict[key][1])
                mitem.set_submenu(menu2)
            elif menu_dict[key][0] == "separator":
                mitem = Gtk.SeparatorMenuItem()
            else:
                if menu_dict[key][1] != None:
                    mitem.connect("activate", menu_dict[key][1])
                else:
                    pass
            menu.append(mitem)
        return menu
    
    def _build_glmenu_from_dicts(self, menu_dict, glMenu):
        """ Function doc """
        for key in menu_dict:
            mitem = Gtk.MenuItem(label=key)
            if menu_dict[key][0] == "submenu":
                menu2 = self._build_submenus_from_dicts(menu_dict[key][1])
                mitem.set_submenu(menu2)
            elif menu_dict[key][0] == "separator":
                mitem = Gtk.SeparatorMenuItem()
            else:
                if menu_dict[key][1] != None:
                    mitem.connect("activate", menu_dict[key][1])
                else:
                    pass
            glMenu.append(mitem)
    
    def save_image(self, filename):
        self.make_current()

        if self.get_error() is not None:
            print("Erro no contexto OpenGL")
            return

        width = self.get_allocated_width()
        height = self.get_allocated_height()

        # Ler pixels
        #glReadBuffer(GL_BACK)
        
        data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)

        # Converter para numpy
        image = np.frombuffer(data, dtype=np.uint8)
        image = image.reshape((height, width, 4))

        # Inverter verticalmente (OpenGL -> imagem)
        image = np.flip(image, axis=0)

        # Criar Pixbuf
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            image.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            True,
            8,
            width,
            height,
            width * 4
        )


        #img = Image.fromarray(image)
        #img = img.filter(ImageFilter.GaussianBlur(1))
        open_preview(image)
        
        
        '''
        #img.filter(ImageFilter.EDGE_ENHANCE)
        img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        img.save("saida.png")
        
        #              efeito cartoon
        # 1. suavizar
        smooth = img.filter(ImageFilter.GaussianBlur(0.2))
        # 2. reduzir cores
        quant = smooth.quantize(colors=32).convert("RGB")
        # 3. detectar bordas
        edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.invert(edges)
        # 4. combinar
        cartoon = Image.blend(quant, edges.convert("RGB"), 0.8)
        cartoon.save("cartoon.png")
        
        
        
        
        #img.save("saida.png")
        
        
        ## Salvar PNG
        #pixbuf.savev(filename, "png", [], [])
        #
        #print(f"Imagem salva em {filename}")
        '''



class PreviewWindow(Gtk.Window):
    def __init__(self, image_array):
        super().__init__(title="Preview Cartoon")

        self.set_default_size(600, 400)

        # imagem original (PIL)
        self.original = Image.fromarray(image_array)

        # preview reduzido (25%)
        self.preview_base = self.original.resize(
            (self.original.width // 4, self.original.height // 4)
        )

        # layout principal
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.add(box)

        # imagem preview
        self.image_widget = Gtk.Image()
        box.pack_start(self.image_widget, True, True, 0)

        # painel de controles
        controls = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.pack_start(controls, False, False, 0)

        # sliders
        self.blur = self.create_slider("Blur", 0.0, 5.0, 1.0, controls)
        self.colors = self.create_slider("Cores", 2, 64, 32, controls)
        self.edge = self.create_slider("Borda", 0, 255, 100, controls)
        self.blend = self.create_slider("Blend", 0.0, 1.0, 0.3, controls)

        # botão salvar
        btn = Gtk.Button(label="Salvar imagem final")
        btn.connect("clicked", self.on_save)
        controls.pack_start(btn, False, False, 0)

        # atualizar preview inicial
        self.update_preview()

    def create_slider(self, label, minv, maxv, default, parent):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        parent.pack_start(box, False, False, 0)

        lbl = Gtk.Label(label=label)
        box.pack_start(lbl, False, False, 0)

        adj = Gtk.Adjustment(default, minv, maxv, 0.1, 1, 0)
        scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=adj)
        scale.connect("value-changed", self.on_change)
        box.pack_start(scale, False, False, 0)

        return scale

    def process_image(self, img):
        blur = self.blur.get_value()
        colors = int(self.colors.get_value())
        edge_th = int(self.edge.get_value())
        blend_val = self.blend.get_value()

        # 1. suavizar
        smooth = img.filter(ImageFilter.GaussianBlur(blur))

        # 2. reduzir cores
        quant = smooth.quantize(colors=colors).convert("RGB")

        # 3. bordas
        edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.invert(edges)
        edges = edges.point(lambda x: 0 if x < edge_th else 255)

        # 4. combinar
        cartoon = Image.blend(quant, edges.convert("RGB"), blend_val)

        return cartoon

    def update_preview(self):
        img = self.process_image(self.preview_base)

        data = np.array(img)
        height, width, _ = data.shape

        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            data.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            width,
            height,
            width * 3
        )

        self.image_widget.set_from_pixbuf(pixbuf)

    def on_change(self, widget):
        self.update_preview()

    def on_save(self, widget):
        final = self.process_image(self.original)
        final.save("cartoon_final.png")
        print("Imagem salva!")



# ---- USO ----
def open_preview(image_array):
    win = PreviewWindow(image_array)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()


