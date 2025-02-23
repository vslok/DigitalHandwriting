using DigitalHandwriting.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Factories.AuthenticationMethods.Models
{
    internal class FilteredManhattanAuthenticationMethod : AuthenticationMethod
    {
        public FilteredManhattanAuthenticationMethod(List<double> userKeyPressedTimes, List<double> userBetweenKeysTimes, List<List<double>> userKeyPressedTimesProfile, List<List<double>> userBetweenKeysTimesProfile) : base(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile)
        {
        }

        public override AuthenticationResult Authenticate(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            if (n > 1)
            {
                return nGraphAuthentication(n, loginKeyPressedTimes, loginBetweenKeysTimes);
            }

            var userProfileData = GetUserProfileData(n);
            var loginKeyPressedNormalized = Calculations.Normalize(loginKeyPressedTimes);
            var loginBetweenKeysNormalized = Calculations.Normalize(loginBetweenKeysTimes);

            var keyPressedDistance = Calculations.ManhattanFilteredDistance(userProfileData[AuthenticationCalculationDataType.H], loginKeyPressedNormalized).distance
                / loginKeyPressedNormalized.Count;
            var betweenKeysDistance = Calculations.ManhattanFilteredDistance(userProfileData[AuthenticationCalculationDataType.UD], loginBetweenKeysNormalized).distance 
                / loginBetweenKeysNormalized.Count;

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

            var userProfileData = GetUserProfileData(n);
            var loginKeyPressedNormalized = Calculations.Normalize(loginKeyPressedTimes);
            var loginBetweenKeysNormalized = Calculations.Normalize(loginBetweenKeysTimes);

            var keyPressedDistance = Calculations.ManhattanFilteredDistance(userProfileData[AuthenticationCalculationDataType.H], loginKeyPressedNormalized).distance
                / loginKeyPressedNormalized.Count;
            var betweenKeysDistance = Calculations.ManhattanFilteredDistance(userProfileData[AuthenticationCalculationDataType.UD], loginBetweenKeysNormalized).distance
                / loginBetweenKeysNormalized.Count;

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
            var userProfileData = GetUserProfileData(n);
            var loginNGraph = Calculations.CalculateNGraph(n, loginKeyPressedTimes, loginBetweenKeysTimes);

            var dataTypeResults = new Dictionary<AuthenticationCalculationDataType, double>();
            foreach (var key in Enum.GetValues(typeof(AuthenticationCalculationDataType)).Cast<AuthenticationCalculationDataType>())
            {
                var values1 = userProfileData[key];
                var values2 = Calculations.Normalize(loginNGraph[key]);

                if (values1[0].Count != values2.Count)
                {
                    throw new ArgumentException("Количество значений для каждой метрики должно быть одинаковым.");
                }

                var metricResult = Calculations.ManhattanFilteredDistance(values1, values2).distance / values2.Count;
                dataTypeResults[key] = metricResult;
            }

            var metricTotal = dataTypeResults.Sum(x => x.Value);
            var authScore = metricTotal / loginNGraph.Keys.Count;
            var isAuthenticated = authScore < 0.15;

            var authResult = new AuthenticationResult(n, dataTypeResults, authScore, isAuthenticated, 0.15);

            return authResult;
        }

        private List<AuthenticationResult> nGraphAuthentication(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes, List<double> thresholds)
        {
            var userProfileData = GetUserProfileData(n);
            var loginNGraph = Calculations.CalculateNGraph(n, loginKeyPressedTimes, loginBetweenKeysTimes);

            var dataTypeResults = new Dictionary<AuthenticationCalculationDataType, double>();
            foreach (var key in Enum.GetValues(typeof(AuthenticationCalculationDataType)).Cast<AuthenticationCalculationDataType>())
            {
                var values1 = userProfileData[key];
                var values2 = Calculations.Normalize(loginNGraph[key]);

                if (values1[0].Count != values2.Count)
                {
                    throw new ArgumentException("Количество значений для каждой метрики должно быть одинаковым.");
                }

                var metricResult = Calculations.ManhattanFilteredDistance(values1, values2);
                dataTypeResults[key] = metricResult.distance / values2.Count;
            }

            var metricTotal = dataTypeResults.Sum(x => x.Value);
            var authScore = metricTotal / loginNGraph.Keys.Count;

            var result = new List<AuthenticationResult>();
            foreach (var threshold in thresholds)
            {
                var isAuthenticated = authScore < threshold;
                var authResult = new AuthenticationResult(n, dataTypeResults, authScore, isAuthenticated, threshold);
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
                    profileData[key].Add(Calculations.Normalize(nGraph[key]));
                }
            }

            return profileData;
        }
    }
}
