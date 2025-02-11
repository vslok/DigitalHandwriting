using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Factories.AuthenticationMethods.Models
{
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

        public abstract bool Authenticate(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes);
    }
}
