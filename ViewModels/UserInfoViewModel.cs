using DigitalHandwriting.Models;
using DigitalHandwriting.Services;
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
        private bool _isAuthenticated = false;

        private double _keyPressedMetric = 0.0;

        private double _betweenKeysMetric = 0.0;

        public UserInfoViewModel(User user, List<int> keyPressedTimes, List<int> beetwenKeysTimes)
        {
            IsAuthentificated = AuthentificationService.HandwritingAuthentification(user, keyPressedTimes, beetwenKeysTimes, 
                out double keyPressedDistance, out double beetweenKeysDistance);

            KeyPressedMetric = keyPressedDistance;
            BetweenKeysMetric = beetweenKeysDistance;
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

        public double BetweenKeysMetric
        {
            get => _betweenKeysMetric;
            set => SetProperty(ref _betweenKeysMetric, value);
        }
    }
}
