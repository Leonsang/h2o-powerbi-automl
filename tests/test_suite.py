import unittest
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_instalacion import TestInstalacion
from tests.test_integracion import TestIntegracion
from tests.test_pbi import TestPowerBI
from tests.test_logger import TestLogger

def suite():
    """Crea y retorna la suite completa de tests"""
    suite = unittest.TestSuite()
    
    # Agregar todos los tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestInstalacion))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegracion))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPowerBI))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLogger))
    
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite()) 