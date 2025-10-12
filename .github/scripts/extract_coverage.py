#!/usr/bin/env python3
"""
Extract coverage percentage from coverage.xml file.
"""
import xml.etree.ElementTree as ET
import os
import sys

def extract_coverage_percentage():
    """Extract coverage percentage from coverage.xml"""
    if not os.path.exists("coverage.xml"):
        return 0
    
    try:
        tree = ET.parse("coverage.xml")
        root = tree.getroot()
        coverage = root.get("line-rate")
        
        if coverage:
            percentage = int(float(coverage) * 100)
            return percentage
        else:
            return 0
    except Exception as e:
        print(f"Error parsing coverage.xml: {e}", file=sys.stderr)
        return 0

if __name__ == "__main__":
    percentage = extract_coverage_percentage()
    print(percentage)
