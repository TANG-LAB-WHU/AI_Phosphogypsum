#!/usr/bin/env python3
"""
Convert CIF (Crystallographic Information File) to XYZ format.

This script reads a CIF file and converts it to XYZ format by:
1. Parsing cell parameters and atomic positions
2. Converting fractional coordinates to Cartesian coordinates
3. Writing the output in XYZ format
"""

import re
import sys
import math
from pathlib import Path


def parse_float_with_uncertainty(value_str):
    """
    Parse a float value that may contain uncertainty in parentheses.
    Example: "5.865(12)" -> 5.865
    """
    if isinstance(value_str, (int, float)):
        return float(value_str)
    
    # Remove uncertainty in parentheses
    match = re.match(r'([+-]?\d+\.?\d*)', str(value_str).strip())
    if match:
        return float(match.group(1))
    return float(value_str)


def parse_cif_file(cif_path):
    """
    Parse a CIF file and extract cell parameters and atomic positions.
    
    Returns:
        dict: Dictionary containing cell parameters and atomic sites
    """
    with open(cif_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    data = {
        'a': None, 'b': None, 'c': None,
        'alpha': None, 'beta': None, 'gamma': None,
        'atoms': []
    }
    
    # Parse cell parameters
    cell_patterns = {
        'a': r'_cell_length_a\s+([^\s]+)',
        'b': r'_cell_length_b\s+([^\s]+)',
        'c': r'_cell_length_c\s+([^\s]+)',
        'alpha': r'_cell_angle_alpha\s+([^\s]+)',
        'beta': r'_cell_angle_beta\s+([^\s]+)',
        'gamma': r'_cell_angle_gamma\s+([^\s]+)',
    }
    
    for key, pattern in cell_patterns.items():
        match = re.search(pattern, content)
        if match:
            data[key] = parse_float_with_uncertainty(match.group(1))
    
    # Parse atomic sites
    # Look for the loop_ section with atom_site data
    atom_site_pattern = r'loop_\s+_atom_site_label\s+_atom_site_type_symbol.*?\n((?:[^\n]+\n)+)'
    match = re.search(atom_site_pattern, content, re.DOTALL)
    
    if match:
        atom_lines = match.group(1).strip().split('\n')
        # Skip header lines and find data lines
        for line in atom_lines:
            line = line.strip()
            if not line or line.startswith('_'):
                continue
            
            # Parse atom line: label symbol multiplicity wyckoff x y z U occupancy
            parts = line.split()
            if len(parts) >= 7:
                label = parts[0]
                symbol = parts[1].rstrip('0123456789+-')  # Remove charge notation
                x_frac = parse_float_with_uncertainty(parts[4])
                y_frac = parse_float_with_uncertainty(parts[5])
                z_frac = parse_float_with_uncertainty(parts[6])
                
                data['atoms'].append({
                    'label': label,
                    'symbol': symbol,
                    'x_frac': x_frac,
                    'y_frac': y_frac,
                    'z_frac': z_frac
                })
    
    return data


def fractional_to_cartesian(x_frac, y_frac, z_frac, a, b, c, alpha, beta, gamma):
    """
    Convert fractional coordinates to Cartesian coordinates.
    
    For orthogonal cells (alpha=beta=gamma=90°), the conversion is simple:
    x_cart = a * x_frac
    y_cart = b * y_frac
    z_cart = c * z_frac
    
    For non-orthogonal cells, we use the transformation matrix.
    """
    # Convert angles to radians
    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)
    gamma_rad = math.radians(gamma)
    
    # Check if cell is orthogonal
    if (abs(alpha - 90.0) < 0.0001 and 
        abs(beta - 90.0) < 0.0001 and 
        abs(gamma - 90.0) < 0.0001):
        # Simple orthogonal conversion
        x_cart = a * x_frac
        y_cart = b * y_frac
        z_cart = c * z_frac
    else:
        # Non-orthogonal cell transformation
        # Calculate cell volume
        volume = a * b * c * math.sqrt(
            1 - math.cos(alpha_rad)**2 - math.cos(beta_rad)**2 - math.cos(gamma_rad)**2
            + 2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad)
        )
        
        # Transformation matrix elements
        a11 = a
        a12 = b * math.cos(gamma_rad)
        a13 = c * math.cos(beta_rad)
        a21 = 0.0
        a22 = b * math.sin(gamma_rad)
        a23 = c * (math.cos(alpha_rad) - math.cos(beta_rad) * math.cos(gamma_rad)) / math.sin(gamma_rad)
        a31 = 0.0
        a32 = 0.0
        a33 = volume / (a * b * math.sin(gamma_rad))
        
        # Apply transformation
        x_cart = a11 * x_frac + a12 * y_frac + a13 * z_frac
        y_cart = a21 * x_frac + a22 * y_frac + a23 * z_frac
        z_cart = a31 * x_frac + a32 * y_frac + a33 * z_frac
    
    return x_cart, y_cart, z_cart


def write_xyz_file(atoms_cartesian, output_path, comment=""):
    """
    Write atoms in Cartesian coordinates to XYZ format.
    
    XYZ format:
    Line 1: Number of atoms
    Line 2: Comment line
    Lines 3+: Atom symbol and x, y, z coordinates
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write number of atoms
        f.write(f"{len(atoms_cartesian)}\n")
        
        # Write comment line
        if comment:
            f.write(f"{comment}\n")
        else:
            f.write("Converted from CIF file\n")
        
        # Write atomic coordinates
        for atom in atoms_cartesian:
            f.write(f"{atom['symbol']:3s} {atom['x']:15.8f} {atom['y']:15.8f} {atom['z']:15.8f}\n")


def main():
    """Main function to convert CIF to XYZ."""
    # Input and output file paths
    script_dir = Path(__file__).parent
    cif_file = script_dir / "conventional_cell.cif"
    xyz_file = script_dir / "conventional_cell.xyz"
    
    # Check if input file exists
    if not cif_file.exists():
        print(f"Error: Input file not found: {cif_file}")
        sys.exit(1)
    
    print(f"Reading CIF file: {cif_file}")
    
    # Parse CIF file
    try:
        cif_data = parse_cif_file(cif_file)
    except Exception as e:
        print(f"Error parsing CIF file: {e}")
        sys.exit(1)
    
    # Validate cell parameters
    required_params = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
    missing_params = [p for p in required_params if cif_data[p] is None]
    if missing_params:
        print(f"Error: Missing cell parameters: {missing_params}")
        sys.exit(1)
    
    if not cif_data['atoms']:
        print("Error: No atomic positions found in CIF file")
        sys.exit(1)
    
    print(f"Found {len(cif_data['atoms'])} atoms")
    print(f"Cell parameters: a={cif_data['a']:.4f}, b={cif_data['b']:.4f}, c={cif_data['c']:.4f}")
    print(f"Cell angles: α={cif_data['alpha']:.2f}°, β={cif_data['beta']:.2f}°, γ={cif_data['gamma']:.2f}°")
    
    # Convert fractional coordinates to Cartesian
    atoms_cartesian = []
    for atom in cif_data['atoms']:
        x_cart, y_cart, z_cart = fractional_to_cartesian(
            atom['x_frac'], atom['y_frac'], atom['z_frac'],
            cif_data['a'], cif_data['b'], cif_data['c'],
            cif_data['alpha'], cif_data['beta'], cif_data['gamma']
        )
        
        atoms_cartesian.append({
            'symbol': atom['symbol'],
            'x': x_cart,
            'y': y_cart,
            'z': z_cart
        })
    
    # Write XYZ file
    comment = f"Converted from {cif_file.name} | Cell: {cif_data['a']:.4f}, {cif_data['b']:.4f}, {cif_data['c']:.4f}, {cif_data['alpha']:.2f}, {cif_data['beta']:.2f}, {cif_data['gamma']:.2f}"
    write_xyz_file(atoms_cartesian, xyz_file, comment)
    
    print(f"Successfully converted to XYZ format: {xyz_file}")
    print(f"Output file contains {len(atoms_cartesian)} atoms")


if __name__ == "__main__":
    main()
