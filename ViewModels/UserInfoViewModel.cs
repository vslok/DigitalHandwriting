using DigitalHandwriting.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Controls;

namespace DigitalHandwriting.ViewModels
{
    public class UserInfoViewModel : BaseViewModel
    {
        private User _user;

        private List<int> _userKeyPressedTimes;

        private List<int> _userBeetwenKeysTimes;

        private List<int> _loginKeyPressedTimes;

        private List<int> _loginBeetwenKeysTimes;

        private bool _isAuthenticated = false;

        private double _keyPressedMetric = 0.0;

        private double _beetwenKeysMetric = 0.0;

        public UserInfoViewModel(User user, List<int> keyPressedTimes, List<int> beetwenKeysTimes)
        {
            _user = user;
            _userKeyPressedTimes = JsonSerializer.Deserialize<List<int>>(_user.KeyPressedTimes);
            _userBeetwenKeysTimes = JsonSerializer.Deserialize<List<int>>(_user.BeetwenKeysTimes);

            _loginKeyPressedTimes = keyPressedTimes;
            _loginBeetwenKeysTimes = beetwenKeysTimes;

            var keyPressedDistance = EuclideanDistance(_userKeyPressedTimes, _loginKeyPressedTimes);
            var beetweenKeysDistance = EuclideanDistance(_userBeetwenKeysTimes, _loginBeetwenKeysTimes);

            IsAuthentificated = keyPressedDistance <= 0.20 && beetweenKeysDistance <= 0.30;
            KeyPressedMetric = keyPressedDistance;
            BeetwenKeysMetric = beetweenKeysDistance;
        }

        public bool IsAuthentificated
        {
            get => _isAuthenticated;
            set => SetProperty(ref _isAuthenticated, value);
        }

        public double KeyPressedMetric
        {
            get => _keyPressedMetric;
            set => SetProperty(ref _keyPressedMetric, value);
        }

        public double BeetwenKeysMetric
        {
            get => _beetwenKeysMetric;
            set => SetProperty(ref _beetwenKeysMetric, value);
        }

        private double EuclideanDistance(List<int> etVector, List<int> curVector)
        {
            var normalizedEtVector = Normalize(etVector);
            var normalizedCurVector = Normalize(curVector);

            var sum = 0.0;
            for (int i = 0; i < normalizedEtVector.Count; i++)
            {
                sum += Math.Pow(normalizedEtVector.ElementAtOrDefault(i) - normalizedCurVector.ElementAtOrDefault(i), 2);
            }
            return Math.Round(Math.Sqrt(sum), 2);
        }

        private List<double> Normalize(List<int> vector)
        {
            var distance = Math.Sqrt(vector.ConvertAll(el => Math.Pow(el, 2)).Sum());
            return vector.ConvertAll(el => el / distance);
        }

    }
}
