"""
Utility script to fix SUMO trips file by adding required attributes
"""

import os
import xml.etree.ElementTree as ET
import argparse
from random import randint

def fix_trips_file(input_file, output_file, scale_factor=1.0, random_depart=False):
    """
    Fix a trips file by adding required ID and depart attributes
    
    Args:
        input_file: Path to input trips file
        output_file: Path to output trips file
        scale_factor: Scale factor for departure times
        random_depart: Whether to use random departure times
    """
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError:
        # Handle case where the file might not be valid XML
        print(f"Could not parse {input_file} as XML. Creating from scratch.")
        content = open(input_file, 'r').read()
        root = ET.Element("trips")
        if content.startswith("<trips>"):
            content = content[7:]  # Remove opening tag
        if content.endswith("</trips>"):
            content = content[:-8]  # Remove closing tag
            
        # Manually parse trips
        for i, trip_str in enumerate(content.split("<trip ")):
            if i == 0:  # Skip first empty split
                continue
                
            # Create a valid XML fragment
            trip_xml = ET.fromstring(f"<trip {trip_str}")
            root.append(trip_xml)
    
    # Process each trip element
    for i, trip in enumerate(root.findall('.//trip')):
        # Add ID if missing
        if 'id' not in trip.attrib:
            trip.set('id', str(i))
            
        # Add depart time if missing
        if 'depart' not in trip.attrib:
            if random_depart:
                depart_time = randint(0, 100) * scale_factor
            else:
                depart_time = i * scale_factor
            trip.set('depart', str(depart_time))
        elif scale_factor != 1.0:
            # Scale existing depart time
            depart_time = float(trip.attrib['depart']) * scale_factor
            trip.set('depart', str(depart_time))
            
        # Rename 'origin' to 'from' if needed
        if 'origin' in trip.attrib:
            trip.set('from', trip.attrib['origin'])
            del trip.attrib['origin']
            
        # Rename 'destination' to 'to' if needed
        if 'destination' in trip.attrib:
            trip.set('to', trip.attrib['destination'])
            del trip.attrib['destination']
    
    # Write fixed file
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Fixed trips file written to {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix SUMO trips file by adding required attributes")
    parser.add_argument("input_file", help="Path to input trips file")
    parser.add_argument("--output-file", "-o", help="Path to output trips file (default: input_file with _fixed suffix)")
    parser.add_argument("--scale-factor", "-s", type=float, default=1.0, help="Scale factor for departure times")
    parser.add_argument("--random-depart", "-r", action="store_true", help="Use random departure times")
    
    args = parser.parse_args()
    
    if args.output_file is None:
        base, ext = os.path.splitext(args.input_file)
        args.output_file = f"{base}_fixed{ext}"
        
    fix_trips_file(args.input_file, args.output_file, args.scale_factor, args.random_depart) 