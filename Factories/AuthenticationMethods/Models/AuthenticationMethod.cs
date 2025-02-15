using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Factories.AuthenticationMethods.Models
{
    public enum AuthenticationCalculationDataType
    {
        H,
        DU,
        UU,
        DD,
    }

    public struct AuthenticationResult
    {
        private readonly int _n;

        private readonly Dictionary<AuthenticationCalculationDataType, double> _dataResults;

        private readonly double _totalAuthenticationScore;

        private readonly bool _isAuthenticated;

        public AuthenticationResult(int n, Dictionary<AuthenticationCalculationDataType, double> dataResults, double totalAuthenticationScore, bool isAuthenticated)
        {
            _n = n;
            _dataResults = dataResults;
            _totalAuthenticationScore = totalAuthenticationScore;
            _isAuthenticated = isAuthenticated;
        }

        public readonly int N => _n;

        public readonly Dictionary<AuthenticationCalculationDataType, double> DataResults => _dataResults;

        public readonly double TotalAuthenticationScore => _totalAuthenticationScore;

        public readonly bool IsAuthenticated => _isAuthenticated;
    }

    public abstract class AuthenticationMethod
    {
        private List<double> _userKeyPressedTimes;

        private List<double> _userBetweenKeysTimes;

        private List<List<double>> _userKeyPressedTimesProfile;

        private List<List<double>> _userBetweenKeysTimesProfile;

        public AuthenticationMethod(
            List<double> userKeyPressedTimes, 
            List<double> userBetweenKeysTimes,
            List<List<double>> userKeyPressedTimesProfile,
            List<List<double>> userBetweenKeysTimesProfile
            )
        {
            _userKeyPressedTimes = userKeyPressedTimes;
            _userBetweenKeysTimes = userBetweenKeysTimes;
            _userKeyPressedTimesProfile = userKeyPressedTimesProfile;
            _userBetweenKeysTimesProfile= userBetweenKeysTimesProfile;
        }

        public List<double> UserKeyPressedTimes => _userKeyPressedTimes;

        public List<double> UserBetweenKeysTimes => _userBetweenKeysTimes;

        public List<List<double>> UserKeyPressedTimesProfile => _userKeyPressedTimesProfile;

        public List<List<double>> UserBetweenKeysTimesProfile => _userBetweenKeysTimesProfile;

        public abstract AuthenticationResult Authenticate(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes);
    }
}
