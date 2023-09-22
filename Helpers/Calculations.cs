using Microsoft.Windows.Themes;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Helpers
{
    public static class Calculations
    {
        public static List<int> CalculateMedianValue(List<List<int>> values)
        {
            var medianValues = new List<int>();
            var rowLength = values.FirstOrDefault(new List<int>()).Count;
            for (int j = 0; j < rowLength; j++)
            {
                var medianValue = 0;
                for (int i = 0; i < values.Count; i++)
                {
                    medianValue += values[i].ElementAtOrDefault(j);
                }
                medianValues.Add(medianValue / values.Count);
            }

            return medianValues;
        }
    }
}
