﻿using DigitalHandwriting.Factories.AuthenticationMethods.Models;
using DigitalHandwriting.Services;
using MathNet.Numerics.Statistics;
using Microsoft.EntityFrameworkCore.Metadata.Conventions;
using Microsoft.EntityFrameworkCore.Metadata.Internal;
using Microsoft.Windows.Themes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace DigitalHandwriting.Helpers
{
    public static class Calculations
    {
        public static List<double> CalculateMedianValue(List<List<double>> values)
        {
            if (values.Count == 0)
                return new List<double>();

            int expectedCount = values[0].Count;
            foreach (var list in values)
            {
                if (list.Count != expectedCount)
                    throw new ArgumentException("All lists must have the same length.");
            }

            List<double> medians = new List<double>();

            for (int i = 0; i < expectedCount; i++)
            {
                var column = values.Select(list => list[i]).OrderBy(x => x).ToList();
                int n = column.Count;

                if (n % 2 == 1)
                {
                    medians.Add(column[n / 2]);
                }
                else
                {
                    medians.Add((column[(n / 2) - 1] + column[n / 2]) / 2.0);
                }
            }

            return medians;
        }

        public static List<double> CalculateMeanValue(List<List<double>> values)
        {
            foreach (var list in values)
            {
                if (list.Count != values[0].Count)
                {
                    throw new Exception("Mean value cannot be calculated");
                }
            }

            List<double> meansList = new List<double>();

            for (int i = 0; i < values[0].Count; i++)
            {
                var valuesList = values.Select(list => list[i]);
                meansList.Add(valuesList.Average());
            }

            return meansList;
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
                score += diff * diff;
            }

            return Math.Round(Math.Sqrt(score), 3);
        }

        public static double NormalizedEuclideanDistance(List<double> vector1, List<double> vector2)
        {
            if (vector1.Count != vector2.Count)
            {
                throw new ArgumentException("Vectors must be of same length.");
            }

            double squaredDistance = 0.0;
            for (int i = 0; i < vector1.Count; i++)
            {
                double diff = vector1[i] - vector2[i];
                squaredDistance += diff * diff;
            }

            double vectorNorm = Math.Sqrt(vector1.Sum(x => x * x));
            double meanVectorNorm = Math.Sqrt(vector2.Sum(x => x * x));

            return squaredDistance / (vectorNorm * meanVectorNorm);
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

            return Math.Round(sumAbs, 3);
        }

        public static (List<double> filteredMeanVector, double distance) ManhattanFilteredDistance(
            List<List<double>> trainData,
            List<double> testVector,
            double stdDeviationThreshold = 3.0)
        {
            if (trainData.Count == 0 || trainData[0].Count != testVector.Count)
            {
                throw new ArgumentException("Invalid input data dimensions");
            }

            var meanVector = CalculateMeanValue(trainData);

            var stdVector = new List<double>();
            for (int j = 0; j < trainData[0].Count; j++)
            {
                var values = trainData.Select(row => row[j]);
                stdVector.Add(Math.Sqrt(values.Select(x => Math.Pow(x - meanVector[j], 2)).Average()));
            }

            var filteredTrainData = trainData.Where(row =>
            {
                return !row.Select((val, idx) => Math.Abs(val - meanVector[idx]) > stdDeviationThreshold * stdVector[idx])
                        .Any(isOutlier => isOutlier);
            }).ToList();

            var filteredMeanVector = CalculateMeanValue(filteredTrainData);

            var distance = ManhattanDistance(testVector, filteredMeanVector);

            return (filteredMeanVector, distance);
        }

        public static double ScaledManhattanDistance(
            List<List<double>> trainData,
            List<double> testVector)
        {
            if (trainData.Count == 0 || trainData[0].Count != testVector.Count)
                throw new ArgumentException("Invalid input data.");

            List<double> medianVector = CalculateMedianValue(trainData);
            List<double> madVector = new List<double>();

            for (int j = 0; j < medianVector.Count; j++)
            {
                var deviations = trainData.Select(row => Math.Abs(row[j] - medianVector[j])).ToList();
                deviations.Sort();
                madVector.Add(deviations.Median());
            }

            double distance = 0;
            for (int j = 0; j < testVector.Count; j++)
            {
                double diff = Math.Abs(testVector[j] - medianVector[j]);
                double denominator = madVector[j] < 0.0001 ? 1.0 : madVector[j];
                distance += diff / denominator;
            }

            return Math.Round(distance, 3);
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

        public static double ITAD(List<List<double>> profile, List<double> current)
        {
            if (profile.Count == 0 || profile.Any(p => p == null || p.Count != current.Count))
                return 0;

            int numKeys = current.Count;
            double total = 0;

            for (int keyIndex = 0; keyIndex < numKeys; keyIndex++)
            {
                var keyMeasurements = profile.Select(p => p[keyIndex]).OrderBy(x => x).ToList();
                double median = keyMeasurements.Median();

                double currentValue = current[keyIndex];

                double cdf = CalculateCDF(keyMeasurements, currentValue);
                total += currentValue <= median ? cdf : 1 - cdf;
            }

            return total / numKeys;
        }

        public static List<List<T>> MatrixTransposing<T>(List<List<T>> matrix)
        {
            return matrix.SelectMany((l, i) => l.Select((d, j) => new { i, j, d }))
                         .GroupBy(l => l.j)
                         .Select(l => l.Select(ll => ll.d).ToList())
                         .ToList();
        }

        public static Dictionary<AuthenticationCalculationDataType, List<double>> CalculateNGraph(int n, List<double> holdTimes, List<double> betweenKeysTimes)
        {
            if (n == 1)
            {
                return new Dictionary<AuthenticationCalculationDataType, List<double>>() {
                    { AuthenticationCalculationDataType.H, holdTimes },
                    { AuthenticationCalculationDataType.UD, betweenKeysTimes }
                };
            }

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
                var nGraphBetweenKeysDownValues = new List<double>();
                var nGraphBetweenKeysUpValues = new List<double>();
                var nGraphHoldValues = new List<double>();
                var nGraphBetweenKeysValues = new List<double>();

                for (int j = 0; j < n - 1; j++)
                {
                    nGraphBetweenKeysDownValues.Add(holdTimes[i + j]);
                    nGraphBetweenKeysDownValues.Add(betweenKeysTimes[i + j]);

                    nGraphBetweenKeysUpValues.Add(betweenKeysTimes[i + j]);
                    nGraphBetweenKeysUpValues.Add(holdTimes[i + j + 1]);

                    nGraphBetweenKeysValues.Add(betweenKeysTimes[i + j]);
                }

                for (int j = 0; j < n; j++)
                {
                    nGraphHoldValues.Add(holdTimes[i + j]);
                }

                nGraphBetweenKeysDown.Add(nGraphBetweenKeysDownValues.Sum());
                nGraphBetweenKeysUp.Add(nGraphBetweenKeysUpValues.Sum());
                nGraphHold.Add(nGraphHoldValues.Sum());
                nGraphBetweenKeys.Add(nGraphBetweenKeysValues.Sum());
            }

            return new Dictionary<AuthenticationCalculationDataType, List<double>>
            {
                { AuthenticationCalculationDataType.H, nGraphHold },
                { AuthenticationCalculationDataType.DD, nGraphBetweenKeysDown },
                { AuthenticationCalculationDataType.UU, nGraphBetweenKeysUp },
                { AuthenticationCalculationDataType.UD, nGraphBetweenKeys },
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
                { AuthenticationCalculationDataType.UD, new List<List<double>>() },
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

        private static double CalculateCDF(List<double> sorted, double value)
        {
            int n = sorted.Count;
            if (n == 0)
            {
                throw new ArgumentException("Список данных не может быть пустым.");
            }

            if (value < sorted[0])
            {
                return 0.0;
            }
            if (value >= sorted[n - 1])
            {
                return 1.0;
            }

            int index = sorted.BinarySearch(value);
            if (index < 0)
            {
                index = ~index - 1;
            }
            else
            {
                while (index < n - 1 && sorted[index + 1] == value)
                {
                    index++;
                }
            }

            double lower = sorted[index];
            double upper = (index < n - 1) ? sorted[index + 1] : lower;

            if (upper == lower)
            {
                return (index + 1) / (double)n;
            }

            double fraction = (value - lower) / (upper - lower);
            double cdfLower = (index + 1) / (double)n;
            double cdfUpper = (index + 2) / (double)n;

            return cdfLower + fraction * (cdfUpper - cdfLower);
        }

        public static class BiometricMetrics
        {
            public class ThresholdMetrics
            {
                public double FAR { get; set; }
                public double FRR { get; set; }
                public double Accuracy { get; set; }
                public double ErrorRate { get; set; }
                public double Precision { get; set; }
                public double Recall { get; set; }
                public double FMeasure { get; set; }
            }

            public static (double EER, double EERThreshold, Dictionary<double, ThresholdMetrics> ThresholdMetrics)
                CalculateMetrics(IEnumerable<CsvExportAuthentication> results)
            {
                var legalUsers = results.Where(r => r.IsLegalUser).ToList();
                var impostors = results.Where(r => !r.IsLegalUser).ToList();

                if (legalUsers.Count == 0 || impostors.Count == 0)
                {
                    throw new ArgumentException("Both legal users and impostors data must be present");
                }

                var userScores = legalUsers.Select(r => r.TotalAuthenticationScore).ToList();
                var impostersScores = impostors.Select(r => r.TotalAuthenticationScore).ToList();
                var allScores = userScores.Concat(impostersScores).OrderBy(s => s).ToList();

                var thresholds = allScores.Distinct().OrderBy(s => s).ToList();
                var frrs = new List<double>();
                var fars = new List<double>();

                foreach (var threshold in thresholds)
                {
                    double falseRejections = userScores.Count(s => s >= threshold);
                    double frr = falseRejections / userScores.Count;

                    double falseAcceptances = impostersScores.Count(s => s < threshold);
                    double far = falseAcceptances / impostersScores.Count;

                    frrs.Add(frr);
                    fars.Add(far);
                }

                double minDiff = double.MaxValue;
                int eerIndex = 0;
                double eer = 0;
                double eerThreshold = 0;

                for (int i = 0; i < thresholds.Count; i++)
                {
                    double diff = Math.Abs(fars[i] - frrs[i]);
                    if (diff < minDiff)
                    {
                        minDiff = diff;
                        eerIndex = i;
                        eer = (fars[i] + frrs[i]) / 2;
                        eerThreshold = thresholds[i];
                    }
                }

                // Calculate metrics for all thresholds
                var allThresholdMetrics = new Dictionary<double, ThresholdMetrics>();
                for (int i = 0; i < thresholds.Count; i++)
                {
                    double threshold = thresholds[i];

                    double truePositives = userScores.Count(s => s < threshold);
                    double falsePositives = impostersScores.Count(s => s < threshold);
                    double trueNegatives = impostersScores.Count(s => s >= threshold);
                    double falseNegatives = userScores.Count(s => s >= threshold);

                    double totalSamples = userScores.Count + impostersScores.Count;

                    double accuracy = ((truePositives + trueNegatives) / totalSamples) * 100;
                    double errorRate = ((falsePositives + falseNegatives) / totalSamples) * 100;

                    double precision = truePositives / (truePositives + falsePositives) * 100;
                    double recall = truePositives / (truePositives + falseNegatives) * 100;
                    double fMeasure = 2 * (precision * recall) / (precision + recall);

                    allThresholdMetrics[threshold] = new ThresholdMetrics
                    {
                        FAR = fars[i] * 100,
                        FRR = frrs[i] * 100,
                        Accuracy = accuracy,
                        ErrorRate = errorRate,
                        Precision = precision,
                        Recall = recall,
                        FMeasure = fMeasure
                    };
                }

                return (eer * 100, eerThreshold, allThresholdMetrics);
            }
        }
    }
}
