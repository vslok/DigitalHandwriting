using DigitalHandwriting.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Factories.AuthenticationMethods.Models
{
    internal class NormalizedManhattanAuthenticationMethod : AuthenticationMethod
    {
        public NormalizedManhattanAuthenticationMethod(List<double> userKeyPressedTimes, List<double> userBetweenKeysTimes, List<List<double>> userKeyPressedTimesProfile, List<List<double>> userBetweenKeysTimesProfile) : base(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile)
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

            var keyPressedDistance = Calculations.ManhattanDistance(userKeyPressedNormalized, loginKeyPressedNormalized);
            var betweenKeysDistance = Calculations.ManhattanDistance(userBetweenKeysNormalized, loginBetweenKeysNormalized);

            var authScore = (keyPressedDistance + betweenKeysDistance) / 2.0;
            var isAuthenticated = authScore < 0.15;

            var authResult = new AuthenticationResult(n, new Dictionary<AuthenticationCalculationDataType, double>()
            {
                { AuthenticationCalculationDataType.H, keyPressedDistance },
                { AuthenticationCalculationDataType.DU, betweenKeysDistance },
            }, authScore, isAuthenticated);

            return authResult;
        }

        private AuthenticationResult nGraphAuthentication(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            var userNGraph = Calculations.CalculateNGraph(n, UserKeyPressedTimes, UserBetweenKeysTimes);
            var loginNGraph = Calculations.CalculateNGraph(n, loginKeyPressedTimes, loginBetweenKeysTimes);

            var dataTypeResults = new Dictionary<AuthenticationCalculationDataType, double>();
            foreach (var key in Enum.GetValues(typeof(AuthenticationCalculationDataType)).Cast<AuthenticationCalculationDataType>())
            {
                var values1 = Calculations.Normalize(userNGraph[key]);
                var values2 = Calculations.Normalize(loginNGraph[key]);

                if (values1.Count != values2.Count)
                {
                    throw new ArgumentException("Количество значений для каждой метрики должно быть одинаковым.");
                }

                var metricResult = Calculations.ManhattanDistance(values1, values2);
                dataTypeResults[key] = metricResult;
            }

            var metricTotal = dataTypeResults.Sum(x => x.Value);
            var authScore = metricTotal / userNGraph.Keys.Count;
            var isAuthenticated = authScore < 0.15;

            var authResult = new AuthenticationResult(n, dataTypeResults, authScore, isAuthenticated);

            return authResult;
        }
    }
}
