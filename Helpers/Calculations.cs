using DigitalHandwriting.Factories.AuthenticationMethods.Models;
using Microsoft.EntityFrameworkCore.Metadata.Conventions;
using Microsoft.Windows.Themes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

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

        public static List<double> CalculateMedianValue(List<List<double>> values)
        {
            foreach (var list in values)
            {
                if (list.Count != values[0].Count)
                {
                    throw new Exception("Median value cannot be calculated");
                }
            }

            List<double> mediansList = new List<double>();

            for (int i = 0; i < values[0].Count; i++)
            {
                var valuesList = values.Select(list => list[i]).OrderBy(x => x).ToList();
                mediansList.Add(valuesList.ElementAt((int)Math.Ceiling(valuesList.Count / 2.0d)));
            }

            return mediansList;
        }

        public static double EuclideanDistance(List<double> vector1, List<double> vector2)
        {
            if (vector1.Count != vector2.Count)
            {
                throw new ArgumentException("Vectors must be of same length.");
            }

            double score = 0.0;
            for (int i = 0; i < vector1.Count; i++)
            {
                double diff = vector1[i] - vector2[i];
                score += Math.Sqrt(diff * diff);
            }

            return Math.Round(score / vector1.Count, 3);
        }

        public static double ManhattanDistance(List<double> vector1, List<double> vector2)
        {
            if (vector1.Count != vector2.Count)
            {
                throw new ArgumentException("Vectors must be of same length.");
            }

            double sumAbs = 0.0;
            for (int i = 0; i < vector1.Count; i++)
            {
                sumAbs += Math.Abs(vector1[i] - vector2[i]);
            }

            return Math.Round(sumAbs / vector1.Count, 3);
        }

        public static double ManhattanDistance(double etPoint, double curPoint)
        { 
            return Math.Round(Math.Abs(etPoint - curPoint), 3);
        }

        public static double MahalanobisDistance(List<double> vector1, List<double> vector2, double[,] invCovMatrix)
        {
            if (vector1.Count != vector2.Count)
            {
                throw new ArgumentException("Vectors must be of same length.");
            }

            int n = vector1.Count;
            double[] delta = new double[n];
            for (int i = 0; i < n; i++)
            {
                delta[i] = vector1[i] - vector2[i];
            }

            // Perform delta^T * invCovMatrix * delta
            double[] temp = new double[n];
            for (int i = 0; i < n; i++)
            {
                temp[i] = 0.0;
                for (int j = 0; j < n; j++)
                {
                    temp[i] += invCovMatrix[i, j] * delta[j];
                }
            }

            double mahalanobis = 0.0;
            for (int i = 0; i < n; i++)
            {
                mahalanobis += delta[i] * temp[i];
            }

            return Math.Sqrt(mahalanobis);
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
            var length = Math.Sqrt(vector.ConvertAll(el => Math.Pow(el, 2)).Sum());
            return vector.ConvertAll(el => el / length);
        }

        public static List<double> Normalize(List<double> vector)
        {
            var length = Math.Sqrt(vector.ConvertAll(el => Math.Pow(el, 2)).Sum());
            return vector.ConvertAll(el => el / length);
        }

        public static double ITAD(List<double> sample, List<List<double>> rawReferences)
        {
            var totalITAD = 0.0;
            for (var i = 0; i < rawReferences[0].Count; i++)
            {
                var keyValues = new List<double>();
                for (var j = 0; j < rawReferences.Count; j++)
                {
                    keyValues.Add(rawReferences[j][i]);
                }

                var sortedReference = keyValues.OrderBy(x => x).ToList();
                int count = sortedReference.Count(x => x <= sample[0]);
                double itad = (double)count / sortedReference.Count;
                totalITAD += itad;
            }

            return totalITAD / sample.Count;
        }

        public static List<List<T>> MatrixTransposing<T>(List<List<T>> matrix)
        {
            return matrix.SelectMany((l, i) => l.Select((d, j) => new { i, j, d }))
                         .GroupBy(l => l.j)
                         .Select(l => l.Select(ll => ll.d).ToList())
                         .ToList();
        }

        public static double Expectancy(List<double> intervals)
        {
            return intervals.Average();
        }

        public static double Dispersion(List<double> intervals, double expect)
        {
            return intervals.Select(interv => Math.Pow(interv - expect, 2)).Average();
        }
       
        public static double StandardDeviation(double disp)
        {
            return Math.Sqrt(disp);
        }

        public static Dictionary<AuthenticationCalculationDataType, List<double>> CalculateNGraph(int n, List<double> holdTimes, List<double>betweenKeysTimes)
        {
            if (holdTimes.Count < n)
            {
                throw new ArgumentException("Not enough data to calculate n-graph");
            }

            if (betweenKeysTimes.Count != holdTimes.Count - 1)
            {
                throw new ArgumentException("Not between keys counts doesn't match hold times count");
            }

            List<double> nGraphBetweenKeysDown = new List<double>();
            List<double> nGraphBetweenKeysUp = new List<double>();
            List<double> nGraphHold = new List<double>();
            List<double> nGraphBetweenKeys = new List<double>();

            for (int i = 0; i <= holdTimes.Count - n; i++)
            {
                var nGraphBetweenKeysDownTime = 0.0;
                var nGraphBetweenKeysUpTime = 0.0;
                var nGraphHoldTime = 0.0;
                var nGraphBetweenKeysTime = 0.0;

                for (int j = 0; j < n - 1; j++)
                {
                    nGraphBetweenKeysDownTime += holdTimes[i + j];
                    nGraphBetweenKeysDownTime += betweenKeysTimes[i + j];

                    nGraphBetweenKeysUpTime += betweenKeysTimes[i + j];
                    nGraphBetweenKeysUpTime += holdTimes[i + j + 1];

                    nGraphBetweenKeysTime += betweenKeysTimes[i + j];
                }

                for (int j = 0; j < n; j++)
                {
                    nGraphHoldTime += holdTimes[i + j];
                }

                nGraphBetweenKeysDown.Add(nGraphBetweenKeysDownTime / (n - 1));
                nGraphBetweenKeysUp.Add(nGraphBetweenKeysUpTime / (n - 1));
                nGraphHold.Add(nGraphHoldTime / n);
                nGraphBetweenKeys.Add(nGraphBetweenKeysTime / (n - 1));
            }

            return new Dictionary<AuthenticationCalculationDataType, List<double>>
            {
                { AuthenticationCalculationDataType.H, nGraphHold },
                { AuthenticationCalculationDataType.DD, nGraphBetweenKeysDown },
                { AuthenticationCalculationDataType.UU, nGraphBetweenKeysUp },
                { AuthenticationCalculationDataType.DU, nGraphBetweenKeys },
            };
        }

        public static Dictionary<AuthenticationCalculationDataType, List<List<double>>> CalculateNGraph(int n, List<List<double>> holdTimes, List<List<double>> betweenKeysTimes)
        {
            if (holdTimes.Count != betweenKeysTimes.Count)
            {
                throw new ArgumentException("Not between keys counts doesn't match hold times count");
            }

            var result = new Dictionary<AuthenticationCalculationDataType, List<List<double>>>
            {
                { AuthenticationCalculationDataType.H, new List<List<double>>() },
                { AuthenticationCalculationDataType.DD, new List<List<double>>() },
                { AuthenticationCalculationDataType.UU, new List<List<double>>() },
                { AuthenticationCalculationDataType.DU, new List<List<double>>() },
            };

            for (int i = 0; i < holdTimes.Count; i++)
            {
                var hold = holdTimes[i];
                var between = betweenKeysTimes[i];

                var graph = CalculateNGraph(n, hold, between);

                foreach (var key in graph.Keys)
                {
                    result[key].Add(graph[key]);
                }
            }

            return result;
        }

        public static double GunettiPicardiMetric(Dictionary<AuthenticationCalculationDataType, List<double>> nGraphs1, 
            Dictionary<AuthenticationCalculationDataType, List<double>> nGraphs2, double t, 
            List<string>? sequence1 = null, List<string>? sequence2 = null)
        {
            double aMeasure = CalculateAMeasure(nGraphs1, nGraphs2, t, out var graphsSimilarity);
            if (sequence1 != null && sequence2 != null)
            {
                double rMeasure = CalculateRMeasure(sequence1, sequence2);
                return (aMeasure + rMeasure) / 2.0;
            }

            return aMeasure;
            
        }

        public static double CalculateAMeasure(
            Dictionary<AuthenticationCalculationDataType, List<double>> nGraphs1, 
            Dictionary<AuthenticationCalculationDataType, List<double>> nGraphs2, 
            double t,
            out Dictionary<AuthenticationCalculationDataType, double> graphsSimilarity
            )
        {
            graphsSimilarity = CalculateNGraphsSimilarity(nGraphs1, nGraphs2, t);
            var totalSimilarity = graphsSimilarity.Sum(x => x.Value);
            return 1.0 - totalSimilarity;
        }

        public static double CalculateRMeasure(List<string> sequence1, List<string> sequence2)
        {
            if (sequence1.Count != sequence2.Count)
                throw new ArgumentException("Последовательности должны быть одинаковой длины.");

            double rMeasure = 0;

            for (int i = 0; i < sequence1.Count; i++)
            {
                int posInSequence2 = sequence2.IndexOf(sequence1[i]);

                rMeasure += Math.Abs(i - posInSequence2);
            }

            return rMeasure;
        }

        private static Dictionary<AuthenticationCalculationDataType, double> CalculateNGraphsSimilarity(
            Dictionary<AuthenticationCalculationDataType, List<double>> nGraphs1, Dictionary<AuthenticationCalculationDataType, List<double>> nGraphs2, double t)
        {
            if (!nGraphs1.Keys.SequenceEqual(nGraphs2.Keys))
                throw new ArgumentException("Наборы метрик должны быть одинаковыми.");

            var dataTypeResults = new Dictionary<AuthenticationCalculationDataType, double>();
            /*foreach (var key in Enum.GetValues(typeof(AuthenticationCalculationDataType)).Cast<AuthenticationCalculationDataType>())*/
            foreach (var key in nGraphs1.Keys)
            {
                var values1 = nGraphs1[key];
                var values2 = nGraphs2[key];

                if (values1.Count != values2.Count)
                    throw new ArgumentException("Количество значений для каждой метрики должно быть одинаковым.");

                var similarCount = 0;
                for (int i = 0; i < values1.Count; i++)
                {
                    double value1 = values1[i];
                    double value2 = values2[i];

                    if (Math.Max(value1, value2) / Math.Min(value1, value2) <= t)
                    {
                        similarCount++;
                    }
                }

                dataTypeResults.Add(key, similarCount / values1.Count);
            }

            return dataTypeResults;
        }
    }
}
