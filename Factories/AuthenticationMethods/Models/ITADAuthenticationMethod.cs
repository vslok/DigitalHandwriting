using DigitalHandwriting.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Factories.AuthenticationMethods.Models
{
    internal class ITADAuthenticationMethod : AuthenticationMethod
    {
        public ITADAuthenticationMethod(List<double> userKeyPressedTimes, List<double> userBetweenKeysTimes, List<List<double>> userKeyPressedTimesProfile, List<List<double>> userBetweenKeysTimesProfile) : base(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile)
        {
        }

        public override AuthenticationResult Authenticate(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            if (n > 1)
            {
                return nGraphAuthentication(n, loginKeyPressedTimes, loginBetweenKeysTimes);
            }

            // For n=1, UserKeyPressedTimesProfile and UserBetweenKeysTimesProfile are the raw profile data.
            // Calculations.ITAD expects List<List<double>> for profile.
            var keyPressedSimilarity = Calculations.ITAD(UserKeyPressedTimesProfile, loginKeyPressedTimes);
            var betweenKeysSimilarity = Calculations.ITAD(UserBetweenKeysTimesProfile, loginBetweenKeysTimes);

            // Average similarity
            var avgSimilarity = (keyPressedSimilarity + betweenKeysSimilarity) / 2.0;

            // Convert to distance-like score (0 is best, 0.5 is worst)
            var authScore = 0.5 - avgSimilarity;
            var isAuthenticated = authScore < 0.05; // Adjusted threshold for distance (0.5 - 0.45)

            var authResult = new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>()
            {
                // Store individual distance-like scores if needed, or original similarities.
                // For now, let's store transformed scores for consistency in what DataResults represents.
                { AuthenticationCalculationDataType.H, 0.5 - keyPressedSimilarity },
                { AuthenticationCalculationDataType.UD, 0.5 - betweenKeysSimilarity },
            }, authScore, isAuthenticated, 0.05); // Adjusted threshold

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

            var keyPressedSimilarity = Calculations.ITAD(UserKeyPressedTimesProfile, loginKeyPressedTimes);
            var betweenKeysSimilarity = Calculations.ITAD(UserBetweenKeysTimesProfile, loginBetweenKeysTimes);

            var avgSimilarity = (keyPressedSimilarity + betweenKeysSimilarity) / 2.0;
            var authScore = 0.5 - avgSimilarity; // Convert to distance-like score

            var result = new List<AuthenticationResult>();
            foreach (var threshold in thresholds) // Assuming these thresholds are for distance scores
            {
                var isAuthenticated = authScore < threshold;
                var authResult = new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>()
                {
                    { AuthenticationCalculationDataType.H, 0.5 - keyPressedSimilarity },
                    { AuthenticationCalculationDataType.UD, 0.5 - betweenKeysSimilarity },
                }, authScore, isAuthenticated, threshold);
                result.Add(authResult);
            }

            return result;
        }

        private AuthenticationResult nGraphAuthentication(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            var userNGraphProfileData = GetUserProfileData(n);
            var loginNGraphFeatures = Calculations.CalculateNGraph(n, loginKeyPressedTimes, loginBetweenKeysTimes);

            var dataTypeSimilarityResults = new Dictionary<AuthenticationCalculationDataType, double>();
            var dataTypeDistanceResults = new Dictionary<AuthenticationCalculationDataType, double>();

            foreach (var key in userNGraphProfileData.Keys)
            {
                if (loginNGraphFeatures.ContainsKey(key) && userNGraphProfileData[key].Any())
                {
                    var similarity = Calculations.ITAD(userNGraphProfileData[key], loginNGraphFeatures[key]);
                    dataTypeSimilarityResults[key] = similarity;
                    dataTypeDistanceResults[key] = 0.5 - similarity;
                }
            }

            if (!dataTypeDistanceResults.Any())
            {
                 return new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>(), 0.5, false, 0.05); // Max distance, not authenticated
            }

            var totalDistance = dataTypeDistanceResults.Sum(x => x.Value);
            var authScore = totalDistance / dataTypeDistanceResults.Count;
            var isAuthenticated = authScore < 0.05;

            var authResult = new AuthenticationResult(n, dataTypeDistanceResults, authScore, isAuthenticated, 0.05);
            return authResult;
        }

        private List<AuthenticationResult> nGraphAuthentication(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes, List<double> thresholds)
        {
            var userNGraphProfileData = GetUserProfileData(n);
            var loginNGraphFeatures = Calculations.CalculateNGraph(n, loginKeyPressedTimes, loginBetweenKeysTimes);

            var dataTypeSimilarityResults = new Dictionary<AuthenticationCalculationDataType, double>();
            var dataTypeDistanceResults = new Dictionary<AuthenticationCalculationDataType, double>();

            foreach (var key in userNGraphProfileData.Keys)
            {
                if (loginNGraphFeatures.ContainsKey(key) && userNGraphProfileData[key].Any())
                {
                    var similarity = Calculations.ITAD(userNGraphProfileData[key], loginNGraphFeatures[key]);
                    dataTypeSimilarityResults[key] = similarity;
                    dataTypeDistanceResults[key] = 0.5 - similarity;
                }
            }

            if (!dataTypeDistanceResults.Any())
            {
                var emptyResults = new List<AuthenticationResult>();
                foreach (var threshold_dist in thresholds) // Assuming thresholds are distance-based
                {
                    emptyResults.Add(new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>(), 0.5, false, threshold_dist));
                }
                return emptyResults;
            }

            var totalDistance = dataTypeDistanceResults.Sum(x => x.Value);
            var authScore = totalDistance / dataTypeDistanceResults.Count;

            var result = new List<AuthenticationResult>();
            foreach (var threshold_dist in thresholds) // Assuming thresholds are distance-based
            {
                var isAuthenticated = authScore < threshold_dist;
                var authResult = new AuthenticationResult(n, dataTypeDistanceResults, authScore, isAuthenticated, threshold_dist);
                result.Add(authResult);
            }

            return result;
        }

        private Dictionary<AuthenticationCalculationDataType, List<List<double>>> GetUserProfileData(int n)
        {
            var profileData = new Dictionary<AuthenticationCalculationDataType, List<List<double>>>();

            if (UserKeyPressedTimesProfile.Count != UserBetweenKeysTimesProfile.Count)
            {
                throw new ArgumentException("Inconsistent profile data");
            }

            for (int i = 0; i < UserKeyPressedTimesProfile.Count; i++)
            {
                var nGraph = Calculations.CalculateNGraph(n, UserKeyPressedTimesProfile[i], UserBetweenKeysTimesProfile[i]);

                foreach (var key in nGraph.Keys)
                {
                    if (!profileData.ContainsKey(key))
                    {
                        profileData[key] = new List<List<double>>();
                    }
                    profileData[key].Add(nGraph[key]);
                }
            }

            return profileData;
        }
    }
}
