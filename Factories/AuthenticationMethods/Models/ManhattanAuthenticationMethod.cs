using DigitalHandwriting.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Factories.AuthenticationMethods.Models
{
    internal class ManhattanAuthenticationMethod : AuthenticationMethod
    {
        public ManhattanAuthenticationMethod(List<double> userKeyPressedTimes, List<double> userBetweenKeysTimes, List<List<double>> userKeyPressedTimesProfile, List<List<double>> userBetweenKeysTimesProfile) : base(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile)
        {
        }

        public override AuthenticationResult Authenticate(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            if (n > 1)
            {
                return nGraphAuthentication(n, loginKeyPressedTimes, loginBetweenKeysTimes);
            }

            var userKeyPressedNormalized = Calculations.Normalize(UserKeyPressedTimes);
            var loginKeyPressedNormalized = Calculations.Normalize(loginKeyPressedTimes);
            var userBetweenKeysNormalized = Calculations.Normalize(UserBetweenKeysTimes);
            var loginBetweenKeysNormalized = Calculations.Normalize(loginBetweenKeysTimes);

            var keyPressedDistance = Calculations.ManhattanDistance(userKeyPressedNormalized, loginKeyPressedNormalized) / loginKeyPressedNormalized.Count;
            var betweenKeysDistance = Calculations.ManhattanDistance(userBetweenKeysNormalized, loginBetweenKeysNormalized) / loginBetweenKeysNormalized.Count;

            var authScore = (keyPressedDistance + betweenKeysDistance) / 2.0;
            var isAuthenticated = authScore < 0.15;

            var authResult = new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>()
            {
                { AuthenticationCalculationDataType.H, keyPressedDistance },
                { AuthenticationCalculationDataType.UD, betweenKeysDistance },
            }, authScore, isAuthenticated, 0.15);

            return authResult;
        }

        public override List<AuthenticationResult> Authenticate(
            int n, List<double> loginKeyPressedTimes,
            List<double> loginBetweenKeysTimes,
            List<double> thresholds)
        {
            if (n > 1)
            {
                return nGraphAuthentication(n, loginKeyPressedTimes, loginBetweenKeysTimes, thresholds);
            }

            var userKeyPressedNormalized = Calculations.Normalize(UserKeyPressedTimes);
            var loginKeyPressedNormalized = Calculations.Normalize(loginKeyPressedTimes);
            var userBetweenKeysNormalized = Calculations.Normalize(UserBetweenKeysTimes);
            var loginBetweenKeysNormalized = Calculations.Normalize(loginBetweenKeysTimes);

            var keyPressedDistance = Calculations.ManhattanDistance(userKeyPressedNormalized, loginKeyPressedNormalized) / loginKeyPressedNormalized.Count;
            var betweenKeysDistance = Calculations.ManhattanDistance(userBetweenKeysNormalized, loginBetweenKeysNormalized) / loginBetweenKeysNormalized.Count;

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

        private AuthenticationResult nGraphAuthentication(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
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
                        var metricResult = Calculations.ManhattanDistance(meanUserNGraphFeatureVectorForKey, loginNGraph[key]);
                        dataTypeResults[key] = metricResult;
                    }
                }
            }

            if (!dataTypeResults.Any())
            {
                return new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>(), double.MaxValue, false, 0.15);
            }

            var metricTotal = dataTypeResults.Sum(x => x.Value);
            var authScore = metricTotal / dataTypeResults.Count;
            var isAuthenticated = authScore < 0.15;

            var authResult = new AuthenticationResult(n, dataTypeResults, authScore, isAuthenticated, 0.15);
            return authResult;
        }

        private List<AuthenticationResult> nGraphAuthentication(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes, List<double> thresholds)
        {
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
                        var metricResult = Calculations.ManhattanDistance(meanUserNGraphFeatureVectorForKey, loginNGraph[key]);
                        dataTypeResults[key] = metricResult;
                    }
                }
            }

            if (!dataTypeResults.Any())
            {
                var emptyResults = new List<AuthenticationResult>();
                foreach (var threshold in thresholds)
                {
                    emptyResults.Add(new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>(), double.MaxValue, false, threshold));
                }
                return emptyResults;
            }

            var metricTotal = dataTypeResults.Sum(x => x.Value);
            var authScore = metricTotal / dataTypeResults.Count;

            var result = new List<AuthenticationResult>();
            foreach (var threshold in thresholds)
            {
                var isAuthenticated = authScore < threshold;
                var authResult = new AuthenticationResult(n, dataTypeResults, authScore, isAuthenticated, threshold);
                result.Add(authResult);
            }

            return result;
        }
    }
}
