﻿using DigitalHandwriting.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Factories.AuthenticationMethods.Models
{
    class EuclidianAuthenticationMethod : AuthenticationMethod
    {
        public EuclidianAuthenticationMethod(List<double> userKeyPressedTimes, List<double> userBetweenKeysTimes, List<List<double>> userKeyPressedTimesProfile, List<List<double>> userBetweenKeysTimesProfile) : base(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile)
        {
        }

        public override async Task<AuthenticationResult> Authenticate(string login, int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            // The 'login' parameter is not used in this specific local authentication method.
            // It's added for conformity with the base class, which now supports ML service calls.
            await Task.CompletedTask; // To make the method async without changing its core logic

            if (n > 1)
            {
                return nGraphAuthentication(login, n, loginKeyPressedTimes, loginBetweenKeysTimes);
            }

            var keyPressedDistance = Calculations.EuclideanDistance(UserKeyPressedTimes, loginKeyPressedTimes);
            var betweenKeysDistance = Calculations.EuclideanDistance(UserBetweenKeysTimes, loginBetweenKeysTimes);

            var authScore = (keyPressedDistance + betweenKeysDistance) / 2.0;
            var isAuthenticated = authScore < 0.15;

            var authResult = new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>()
            {
                { AuthenticationCalculationDataType.H, keyPressedDistance },
                { AuthenticationCalculationDataType.UD, betweenKeysDistance },
            }, authScore, isAuthenticated, 0.15);

            return authResult;
        }

        public override async Task<List<AuthenticationResult>> Authenticate(
            string login, int n, List<double> loginKeyPressedTimes,
            List<double> loginBetweenKeysTimes,
            List<double> thresholds)
        {
            // The 'login' parameter is not used in this specific local authentication method.
            await Task.CompletedTask; // To make the method async

            if (n > 1)
            {
                return nGraphAuthentication(login, n, loginKeyPressedTimes, loginBetweenKeysTimes, thresholds);
            }

            var keyPressedDistance = Calculations.EuclideanDistance(UserKeyPressedTimes, loginKeyPressedTimes);
            var betweenKeysDistance = Calculations.EuclideanDistance(UserBetweenKeysTimes, loginBetweenKeysTimes);

            var authScore = (keyPressedDistance + betweenKeysDistance) / 2.0;

            var result = new List<AuthenticationResult>();
            foreach (var threshold in thresholds)
            {
                var isAuthenticated = authScore < threshold;
                var authResult = new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>()
                {
                    { AuthenticationCalculationDataType.H, keyPressedDistance },
                    { AuthenticationCalculationDataType.UD, betweenKeysDistance },
                }, authScore, isAuthenticated, threshold);
                result.Add(authResult);
            }

            return result;
        }

        private AuthenticationResult nGraphAuthentication(string login, int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            // 'login' parameter added for conformity, not used here
            // Calculate n-graphs for each sample in the user's profile
            var userNGraphProfile = Calculations.CalculateNGraph(n, UserKeyPressedTimesProfile, UserBetweenKeysTimesProfile);
            var loginNGraph = Calculations.CalculateNGraph(n, loginKeyPressedTimes, loginBetweenKeysTimes);

            var dataTypeResults = new Dictionary<AuthenticationCalculationDataType, double>();

            foreach (var key in userNGraphProfile.Keys)
            {
                if (loginNGraph.ContainsKey(key) && userNGraphProfile[key].Any() && userNGraphProfile[key][0].Any())
                {
                    var meanUserNGraphFeatureVectorForKey = Calculations.CalculateMeanValue(userNGraphProfile[key]);
                    if (meanUserNGraphFeatureVectorForKey.Any())
                    {
                        var metricResult = Calculations.EuclideanDistance(meanUserNGraphFeatureVectorForKey, loginNGraph[key]);
                        dataTypeResults[key] = metricResult;
                    }
                }
            }

            if (!dataTypeResults.Any())
            {
                // Handle case where no common n-graph features could be compared (e.g., due to very short input or profile issues)
                // Returning a result indicating failure or a high (bad) score might be appropriate.
                // For now, creating a result with a high score and not authenticated.
                return new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>(), double.MaxValue, false, 0.15);
            }

            var metricTotal = dataTypeResults.Sum(x => x.Value);
            var authScore = metricTotal / dataTypeResults.Count; // Average over successfully compared feature types
            var isAuthenticated = authScore < 0.15;

            var authResult = new AuthenticationResult(n, dataTypeResults, authScore, isAuthenticated, 0.15);
            return authResult;
        }

        private List<AuthenticationResult> nGraphAuthentication(string login, int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes, List<double> thresholds)
        {
            // 'login' parameter added for conformity, not used here
            // Calculate n-graphs for each sample in the user's profile
            var userNGraphProfile = Calculations.CalculateNGraph(n, UserKeyPressedTimesProfile, UserBetweenKeysTimesProfile);
            var loginNGraph = Calculations.CalculateNGraph(n, loginKeyPressedTimes, loginBetweenKeysTimes);

            var dataTypeResults = new Dictionary<AuthenticationCalculationDataType, double>();

            foreach (var key in userNGraphProfile.Keys)
            {
                if (loginNGraph.ContainsKey(key) && userNGraphProfile[key].Any() && userNGraphProfile[key][0].Any())
                {
                    var meanUserNGraphFeatureVectorForKey = Calculations.CalculateMeanValue(userNGraphProfile[key]);
                    if (meanUserNGraphFeatureVectorForKey.Any())
                    {
                        var metricResult = Calculations.EuclideanDistance(meanUserNGraphFeatureVectorForKey, loginNGraph[key]);
                        dataTypeResults[key] = metricResult;
                    }
                }
            }

            if (!dataTypeResults.Any())
            {
                // Handle case where no common n-graph features could be compared
                var emptyResults = new List<AuthenticationResult>();
                foreach (var threshold in thresholds)
                {
                    emptyResults.Add(new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>(), double.MaxValue, false, threshold));
                }
                return emptyResults;
            }

            var metricTotal = dataTypeResults.Sum(x => x.Value);
            var authScore = metricTotal / dataTypeResults.Count; // Average over successfully compared feature types
            var isAuthenticated = authScore < 0.15;

            var result = new List<AuthenticationResult>();
            foreach (var threshold in thresholds)
            {
                var authResult = new AuthenticationResult(n, dataTypeResults, authScore, isAuthenticated, threshold);
                result.Add(authResult);
            }

            return result;
        }
    }
}
