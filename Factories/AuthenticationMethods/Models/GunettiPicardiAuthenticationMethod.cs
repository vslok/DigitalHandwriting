using DigitalHandwriting.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Factories.AuthenticationMethods.Models
{
    internal class GunettiPicardiAuthenticationMethod : AuthenticationMethod
    {
        public GunettiPicardiAuthenticationMethod(List<double> userKeyPressedTimes, List<double> userBetweenKeysTimes, List<List<double>> userKeyPressedTimesProfile, List<List<double>> userBetweenKeysTimesProfile) : base(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile)
        {
        }

        public override AuthenticationResult Authenticate(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            if (n > 1)
            {
                return nGraphAuthentication(n, loginKeyPressedTimes, loginBetweenKeysTimes);
            }

            var authScore = Calculations.CalculateAMeasure(new Dictionary<AuthenticationCalculationDataType, List<double>>()
            {
                { AuthenticationCalculationDataType.H, UserKeyPressedTimes },
                { AuthenticationCalculationDataType.DU, UserBetweenKeysTimes },
            }, 
            new Dictionary<AuthenticationCalculationDataType, List<double>>()
            {
                { AuthenticationCalculationDataType.H, loginKeyPressedTimes },
                { AuthenticationCalculationDataType.DU, loginBetweenKeysTimes },
            }, 1.15, out var ngraphsSimiliarity);

            var isAuthenticated = authScore < 0.15;
            var authResult = new AuthenticationResult(n, ngraphsSimiliarity, authScore, isAuthenticated);

            return authResult;
        }

        private AuthenticationResult nGraphAuthentication(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            var userGraph = Calculations.CalculateNGraph(n, UserKeyPressedTimes, UserBetweenKeysTimes);
            var loginGraph = Calculations.CalculateNGraph(n, loginKeyPressedTimes, loginBetweenKeysTimes);

            var authScore = Calculations.CalculateAMeasure(userGraph, loginGraph, 1.15, out var ngraphsSimiliarity);

            var isAuthenticated = authScore < 0.15;
            var authResult = new AuthenticationResult(n, ngraphsSimiliarity, authScore, isAuthenticated);

            return authResult;
        }
    }
}
