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

            var keyPressedDistance = Calculations.ITAD(loginKeyPressedTimes, UserKeyPressedTimesProfile);
            var betweenKeysDistance = Calculations.ITAD(loginBetweenKeysTimes, UserBetweenKeysTimesProfile);

            var authScore = (keyPressedDistance + betweenKeysDistance) / 2.0;
            var isAuthenticated = authScore > 0.45;

            var authResult = new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>()
            {
                { AuthenticationCalculationDataType.H, keyPressedDistance },
                { AuthenticationCalculationDataType.DU, betweenKeysDistance },
            }, authScore, isAuthenticated, 0.45);

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

            var keyPressedDistance = Calculations.ITAD(loginKeyPressedTimes, UserKeyPressedTimesProfile);
            var betweenKeysDistance = Calculations.ITAD(loginBetweenKeysTimes, UserBetweenKeysTimesProfile);

            var authScore = (keyPressedDistance + betweenKeysDistance) / 2.0;

            var result = new List<AuthenticationResult>();
            foreach (var threshold in thresholds)
            {
                var isAuthenticated = authScore > threshold;
                var authResult = new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>()
                {
                    { AuthenticationCalculationDataType.H, keyPressedDistance },
                    { AuthenticationCalculationDataType.DU, betweenKeysDistance },
                }, authScore, isAuthenticated, threshold);
                result.Add(authResult);
            }

            return result;
        }

        private AuthenticationResult nGraphAuthentication(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            var userNGraphs = Calculations.CalculateNGraph(n, UserKeyPressedTimesProfile, UserBetweenKeysTimesProfile);
            var loginNGraph = Calculations.CalculateNGraph(n, loginKeyPressedTimes, loginBetweenKeysTimes);

            var dataTypeResults = new Dictionary<AuthenticationCalculationDataType, double>();
            foreach (var key in Enum.GetValues(typeof(AuthenticationCalculationDataType)).Cast<AuthenticationCalculationDataType>())
            {
                var values1 = userNGraphs[key];
                var values2 = loginNGraph[key];

                if (values1[0].Count != values2.Count)
                {
                    throw new ArgumentException("Количество значений для каждой метрики должно быть одинаковым.");
                }

                var metricResult = Calculations.ITAD(values2, values1);
                dataTypeResults[key] = metricResult;
            }

            var metricTotal = dataTypeResults.Sum(x => x.Value);
            var authScore = metricTotal / loginNGraph.Keys.Count;
            var isAuthenticated = authScore > 0.45;

            var authResult = new AuthenticationResult(n, dataTypeResults, authScore, isAuthenticated, 0.45);
            return authResult;
        }

        private List<AuthenticationResult> nGraphAuthentication(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes, List<double> thresholds)
        {
            var userNGraphs = Calculations.CalculateNGraph(n, UserKeyPressedTimesProfile, UserBetweenKeysTimesProfile);
            var loginNGraph = Calculations.CalculateNGraph(n, loginKeyPressedTimes, loginBetweenKeysTimes);

            var dataTypeResults = new Dictionary<AuthenticationCalculationDataType, double>();
            foreach (var key in Enum.GetValues(typeof(AuthenticationCalculationDataType)).Cast<AuthenticationCalculationDataType>())
            {
                var values1 = userNGraphs[key];
                var values2 = loginNGraph[key];

                if (values1[0].Count != values2.Count)
                {
                    throw new ArgumentException("Количество значений для каждой метрики должно быть одинаковым.");
                }

                var metricResult = Calculations.ITAD(values2, values1);
                dataTypeResults[key] = metricResult;
            }

            var metricTotal = dataTypeResults.Sum(x => x.Value);
            var authScore = metricTotal / loginNGraph.Keys.Count;

            var result = new List<AuthenticationResult>();
            foreach (var threshold in thresholds)
            {
                var isAuthenticated = authScore > threshold;
                var authResult = new AuthenticationResult(n, dataTypeResults, authScore, isAuthenticated, threshold);
                result.Add(authResult);
            }

            return result;
        }
    }
}
