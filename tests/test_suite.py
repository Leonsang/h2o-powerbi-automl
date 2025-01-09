import unittest
import h2o
from tests.test_instalacion import TestInstalacion
from tests.test_integracion import TestIntegracion
from tests.test_pbi import TestPowerBI
from src.logger import Logger
from src.init_h2o_server import iniciar_servidor_h2o

logger = Logger('test_suite')

if __name__ == '__main__':
    logger.info("Iniciando Suite Principal")
    
    # Iniciar H2O antes de los tests
    logger.info("Iniciando servidor H2O...")
    iniciar_servidor_h2o()
    
    # Ejecutar tests
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    
    # Agregar tests en orden
    suite.addTests(loader.loadTestsFromTestCase(TestIntegracion))
    suite.addTests(loader.loadTestsFromTestCase(TestPowerBI))
    suite.addTests(loader.loadTestsFromTestCase(TestInstalacion))
    
    # Ejecutar suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Registrar resultado
    status = 'OK' if result.wasSuccessful() else 'FAIL'
    logger.info(f"Suite Principal: {status}") 