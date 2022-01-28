from Bio.PDB import *
import numpy
import xpdb 
import itertools
import xfel2146_tools

input_file = xfel2146_tools.DATA_DIR / "1SS8_H2O.pdb"
output_dir = xfel2146_tools.BASE_DIR / "Results/Models"
output_dir.mkdir(parents=True, exist_ok=True)

parser = PDBParser(PERMISSIVE=1, structure_builder=xpdb.SloppyStructureBuilder())
structure = parser.get_structure('1SS8', input_file)

water_chain = list(structure.get_chains())[-1]
water_coords = numpy.array([atom.get_coord() for atom in water_chain.get_atoms()])

molecule_atoms = list(itertools.chain(*[list(c.get_atoms()) for c in list(structure.get_chains())[:-1]]))
molecule_coords = numpy.array([atom.get_coord() for atom in molecule_atoms])

center = water_coords.mean(axis=0)

center = numpy.array([ 88.905266, 102.0538  , 140.0412  ])
rot_mat = numpy.array([[ 0.99237907, -0.08058891,  0.09321594],
                       [ 0.08058891,  0.99674029,  0.00377045],
                       [-0.09321594,  0.00377045,  0.99563878]])

def accept_dry(coord):
    return False

def accept_20_cyl(coord):
    length = 135
    radius = 20
    if abs(coord[0]) > length/2:
        return False
    if numpy.sqrt(coord[1]**2 + coord[2]**2) > radius:
        return False
    if numpy.sqrt(coord[0]**2 + coord[1]**2 + coord[2]**2) > numpy.sqrt((length/2)**2 + radius**2) - 14:
        return False
    return True

def accept_35_cyl(coord):
    length = 135
    radius = 35
    if abs(coord[0]) > length/2:
        return False
    if numpy.sqrt(coord[1]**2 + coord[2]**2) > radius:
        return False
    if numpy.sqrt(coord[0]**2 + coord[1]**2 + coord[2]**2) > numpy.sqrt((length/2)**2 + radius**2) - 14:
        return False
    return True

def accept_45_cyl(coord):
    length = 135
    radius = 55
    if abs(coord[0]) > length/2:
        return False
    if numpy.sqrt(coord[1]**2 + coord[2]**2) > radius:
        return False
    if numpy.sqrt(coord[0]**2 + coord[1]**2 + coord[2]**2) > numpy.sqrt((length/2)**2 + radius**2) - 14:
        return False
    return True

def accept_55_cyl(coord):
    length = 135
    radius = 55
    if abs(coord[0]) > length/2:
        return False
    if numpy.sqrt(coord[1]**2 + coord[2]**2) > radius:
        return False
    if numpy.sqrt(coord[0]**2 + coord[1]**2 + coord[2]**2) > length/2 + 5:
        return False
    return True

def accept_65_cyl(coord):
    length = 135
    radius = 65
    if abs(coord[0]) > length/2:
        return False
    if numpy.sqrt(coord[1]**2 + coord[2]**2) > radius:
        return False
    if numpy.sqrt(coord[0]**2 + coord[1]**2 + coord[2]**2) > numpy.sqrt((length/2)**2 + radius**2) - 14:
        return False
    return True

def accept_hollow_cyl(coord):
    length = 135
    inner_radius = 20
    outer_radius = 65
    sphere_radius = numpy.sqrt((length/2)**2 + outer_radius**2) - 14
    if abs(coord[0]) > length/2:
        return False
    if numpy.sqrt(coord[1]**2 + coord[2]**2) > outer_radius:
        return False
    if numpy.sqrt(coord[1]**2 + coord[2]**2) < inner_radius:
        return False
    if numpy.sqrt(coord[0]**2 + coord[1]**2 + coord[2]**2) > sphere_radius:
        return False
    if numpy.sqrt((abs(coord[0])-(length/2+5))**2 + coord[1]**2 + coord[2]**2) < 30:
        return False
    return True


class AtomSelect(Select):
    def __init__(self, accept_function):
        super().__init__()
        self._accept_function = accept_function
        
    def accept_atom(self, atom):
        # Keep the protein
        if(atom.get_parent().get_resname() != 'SOL'):
            return True
        # Kick out the hydrogens
        if(atom.get_name() == 'HW2' or atom.get_name() == 'HW1'):
            return False
        # atom.element = "O"
        coord = rot_mat @ (atom.get_coord() - center)
        return self._accept_function(coord)

io = xpdb.SloppyPDBIO()
io.set_structure(structure)
io.save(str(output_dir / '1SS8.pdb'), AtomSelect(accept_20_cyl))
io.save(str(output_dir / '1SS8_H2O_20_cyl.pdb'), AtomSelect(accept_20_cyl))
io.save(str(output_dir / '1SS8_H2O_35_cyl.pdb'), AtomSelect(accept_35_cyl))
io.save(str(output_dir / '1SS8_H2O_45_cyl.pdb'), AtomSelect(accept_45_cyl))
io.save(str(output_dir / '1SS8_H2O_55_cyl.pdb'), AtomSelect(accept_55_cyl))
io.save(str(output_dir / '1SS8_H2O_65_cyl.pdb'), AtomSelect(accept_65_cyl))
io.save(str(output_dir / '1SS8_H2O_hollow_cyl.pdb'), AtomSelect(accept_hollow_cyl))
