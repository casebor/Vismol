from vismol.utils.debug import dprint
import yaml
import os, pprint
elements_path = os.path.abspath(__file__)
#HOME = os.path.split(HOME)
elements_path, _file = os.path.split(elements_path)
elements_path = os.path.join(elements_path, 'elements')
dprint('\n\n',elements_path)# = os.path.abspath(__dir__)

files  = os.listdir(elements_path)

elements_by_symbol = {}
elements_by_number = {}

# Open the YAML file
for _file in files:
    _file2 = _file.split('.')
    if _file2[1] == 'yaml':
        with open(os.path.join(elements_path,_file), "r") as yamlfile:
            # Load the YAML data
            data = yaml.safe_load(yamlfile)
            elements_by_symbol[_file2[0]] = data
            elements_by_number[data['Atomic Number']] =  data 
# Now 'data' contains the contents of the YAML file
#pprint.pprint(elements_by_number )

for anum in range(1,119):
    # 'Pm' : [ 61,  "Promethium"       ,[0.640000, 1.000000, 0.780000], [ 10, 105,  10], [199, 255, 199],  1.800000 , 1.800000 , 1.700000 ],#                                                          2.05, 0.30, 0.30 ],	
    sym =  '"'+elements_by_number[anum]['Symbol']+'"'
    num  = anum
    name = '"'+elements_by_number[anum]['Name']+'"'
    rgb  = elements_by_number[anum]['RGB Color Indices']
    rgb  = '[{:1.4f} , {:1.4f} , {:1.4f}]'.format(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    mass = elements_by_number[anum]['Mass']
    
    rcov = elements_by_number[anum]['Covalent Radius']
    vdw  = elements_by_number[anum]['vdW Radius']
    
    dprint('{:2} : {:3} , {:15s} , {} , {:9.5f}, {:4.3f} {:4.3f}'.   format(sym, num,name, rgb, mass,  rcov,vdw ))
    
    #try:
    #    print(anum,  
    #          #elements_by_number[anum]['Symbol'],
    #          #'"'+elements_by_number[anum]['Name']+'"',
    #          #'[{:1.4f} , {:1.4f} , {:1.4f}]'.format(rgb[0]/255, rgb[1]/255, rgb[2]/255), #elements_by_number[anum]['RGB Color Indices'],
    #          #elements_by_number[anum]['RGB Color Indices'],
    #          #elements_by_number[anum]['Covalent Radius'],
    #          #elements_by_number[anum]['Bragg Radius'],
    #          #elements_by_number[anum]['vdW Radius'],
    #          elements_by_number[anum]['Mass'],
    #          )
    #except:
    #    print(anum, 'None')

#for symbol in elements_by_symbol.keys():
#    print(symbol, elements_by_symbol[symbol])


'''
# Nome do arquivo YAML
yaml_file = "atom_types.yaml"

# Salvar o dicionário no arquivo YAML
with open(yaml_file, 'w') as file:
    yaml.dump(elements_by_number, file)
'''
