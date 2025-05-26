using DigitalHandwriting.Models;
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
        UD,
        UU,
        DD,
    }

    public class AuthenticationResult
    {
        private readonly int _n;

        private readonly Dictionary<AuthenticationCalculationDataType, double> _dataResults;

        private readonly double _totalAuthenticationScore;

        private readonly double _threshold;

        private readonly bool _isAuthenticated;

        public AuthenticationResult(AuthenticationResult authenticationResult)
        {
            _n = authenticationResult.N;
            _dataResults = authenticationResult.DataResults;
            _totalAuthenticationScore = authenticationResult.TotalAuthenticationScore;
            _isAuthenticated = authenticationResult.IsAuthenticated;
            _threshold = authenticationResult.Threshold;
        }

        public AuthenticationResult(
            int n,
            Dictionary<AuthenticationCalculationDataType, double> dataResults,
            double totalAuthenticationScore,
            bool isAuthenticated,
            double threshold)
        {
            _n = n;
            _dataResults = dataResults;
            _totalAuthenticationScore = totalAuthenticationScore;
            _isAuthenticated = isAuthenticated;
            _threshold = threshold;
        }

        public int N => _n;

        public Dictionary<AuthenticationCalculationDataType, double> DataResults => _dataResults;

        public double TotalAuthenticationScore => _totalAuthenticationScore;

        public bool IsAuthenticated => _isAuthenticated;

        public double Threshold => _threshold;
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

        public abstract Task<List<AuthenticationResult>> Authenticate(
            string login,
            int n,
            List<double> loginKeyPressedTimes,
            List<double> loginBetweenKeysTimes,
            List<double> thresholds);

        public abstract Task<AuthenticationResult> Authenticate(
            string login,
            int n,
            List<double> loginKeyPressedTimes,
            List<double> loginBetweenKeysTimes);
    }
}
