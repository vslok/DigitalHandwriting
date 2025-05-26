using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DigitalHandwriting.Services;

namespace DigitalHandwriting.Factories.AuthenticationMethods.Models
{
    public class LSTMAuthenticationMethod : AuthenticationMethod
    {
        private readonly MLAuthenticationRequestService _mlAuthService;

        public LSTMAuthenticationMethod(
            List<double> userKeyPressedTimes,
            List<double> userBetweenKeysTimes,
            List<List<double>> userKeyPressedTimesProfile,
            List<List<double>> userBetweenKeysTimesProfile)
            : base(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile)
        {
            _mlAuthService = new MLAuthenticationRequestService();
        }

        public override async Task<List<AuthenticationResult>> Authenticate(string login, int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes, List<double> thresholds)
        {
            var singleResult = await Authenticate(login, n, loginKeyPressedTimes, loginBetweenKeysTimes);
            return new List<AuthenticationResult> { singleResult };
        }

        public override async Task<AuthenticationResult> Authenticate(string login, int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            bool isAuthenticated = await _mlAuthService.Authenticate(
                login,
                n,
                loginKeyPressedTimes,
                loginBetweenKeysTimes,
                "LSTM"
            );

            return new AuthenticationResult(
                n,
                new Dictionary<AuthenticationCalculationDataType, double>(),
                0,
                isAuthenticated,
                0
            );
        }
    }
}
