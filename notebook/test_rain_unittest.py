
import unittest
import rain
from rain import df_soil
from rain import ppt
class Testrain(unittest.TestCase):
    def test_findPeaks(self):
        t_swc5 = rain.findPeaks(df_soil.SWC_5[:3000], 0.20, 0.1)
        t_ppt = rain.findPeaks(ppt[:3000], 2, 10)
        self.assertEqual(t_swc5,([2562], [0.223]))
        self.assertEqual(t_ppt,([2560, 2587],[11.43, 15.75]))
    def test_compare(self):
        t_result = rain.Compare(df_soil.SWC_5[:3000],ppt[:3000],0.2,2,0.1,10)
        # actual_result = ([2562],[0.223],[2560,2587],[11.43,15.75])
        self.assertEqual(t_result,([2562],[0.223],[2560, 2587],[11.43, 15.75],100.0,50.0))
    def test_findIncreasing(self):
        t_increase = rain.findIncreasing(df_soil.SWC_5[:100])
        self.assertEqual(t_increase,([48, 49], [0.195, 0.192]))
    def test_Compare2(self):
        t_compare = rain.Compare2(df_soil.SWC_5[:100],ppt[:100])
        self.assertEqual(t_compare, ([48, 49], [0.195, 0.192],100.0))

if __name__ == '__main__':
    unittest.main()