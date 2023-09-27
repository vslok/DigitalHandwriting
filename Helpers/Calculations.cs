using Microsoft.Windows.Themes;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Markup;

namespace DigitalHandwriting.Helpers
{
    public static class Calculations
    {
        public static List<int> CalculateMedianValue(List<List<int>> values)
        {
            foreach (var list in values)
            {
                if (list.Count != values[0].Count)
                {
                    throw new Exception("Median value cannot be calculated");
                }
            }

            List<int> mediansList = new List<int>();

            for (int i = 0; i < values[0].Count; i++)
            {
                var valuesList = values.Select(list => list[i]).OrderBy(x => x).ToList();
                mediansList.Add(valuesList.ElementAt((int)Math.Ceiling(valuesList.Count / 2.0d)));
            }

            return mediansList;
        }

        public static double EuclideanDistance(List<int> etVector, List<int> curVector)
        {
            var normalizedEtVector = Normalize(etVector);
            var normalizedCurVector = Normalize(curVector);

            var sum = 0.0;
            for (int i = 0; i < normalizedEtVector.Count; i++)
            {
                sum += Math.Pow(normalizedEtVector.ElementAtOrDefault(i) - normalizedCurVector.ElementAtOrDefault(i), 2);
            }
            var distance = Math.Sqrt(sum);
            var normalizedDistance = distance / Math.Sqrt(etVector.Count);
            return Math.Round(normalizedDistance, 2);
        }

        public static double ManhattanDistance(List<int> etVector, List<int> curVector)
        {
            var normalizedEtVector = Normalize(etVector);
            var normalizedCurVector = Normalize(curVector);

            double sum = 0;
            for (int i = 0; i < normalizedEtVector.Count; i++)
            {
                sum += Math.Abs(normalizedEtVector[i] - normalizedCurVector[i]);
            }
            return Math.Round(sum / etVector.Count, 2);
        }

        public static double ManhattanDistance(double etPoint, double curPoint)
        { 
            return Math.Round(Math.Abs(etPoint - curPoint), 2);
        }

        public static double CosineSimilarity(List<int> etVector, List<int> curVector)
        {
            double dotProduct = 0;
            double normEnrollment = 0;
            double normCurrent = 0;

            for (int i = 0; i < etVector.Count; i++)
            {
                dotProduct += etVector[i] * curVector[i];
                normEnrollment += Math.Pow(etVector[i], 2);
                normCurrent += Math.Pow(curVector[i], 2);
            }

            return dotProduct / (Math.Sqrt(normEnrollment) * Math.Sqrt(normCurrent));
        }

        public static List<double> Normalize(List<int> vector)
        {
            var distance = Math.Sqrt(vector.ConvertAll(el => Math.Pow(el, 2)).Sum());
            return vector.ConvertAll(el => el / distance);
        }

        public static List<List<T>> MatrixTransposing<T>(List<List<T>> matrix)
        {
            return matrix.SelectMany((l, i) => l.Select((d, j) => new { i, j, d }))
                         .GroupBy(l => l.j)
                         .Select(l => l.Select(ll => ll.d).ToList())
                         .ToList();
        }

        public static double Expectancy(List<int> intervals)
        {
            return intervals.Average();
        }

        public static double Dispersion(List<int> intervals, double expect)
        {
            return intervals.Select(interv => Math.Pow(interv - expect, 2)).Average();
        }
       
        public static double StandardDeviation(double disp)
        {
            return Math.Sqrt(disp);
        }
    }
}
