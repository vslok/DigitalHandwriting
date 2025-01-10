using DigitalHandwriting.Models;
using DigitalHandwriting.Services;
using System.Collections.Generic;

namespace DigitalHandwriting.ViewModels
{
    public class UserInfoViewModel : BaseViewModel
    {
        private bool _isAuthenticated = false;

        private double _keyPressedMetric = 0.0;

        private double _betweenKeysMetric = 0.0;

        private double _betweenKeysPressMetric = 0.0;

        public UserInfoViewModel(bool isAuthenticated, double keyPressedMetric, double betweenKeysMetric, double betweenKeysPressMetric)
        {
            IsAuthentificated = isAuthenticated;
            KeyPressedMetric = keyPressedMetric;
            BetweenKeysMetric = betweenKeysMetric;
            BetweenKeysPressMetric = betweenKeysPressMetric;
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

        public double BetweenKeysPressMetric
        {
            get => _betweenKeysPressMetric;
            set => SetProperty(ref _betweenKeysPressMetric, value);
        }
    }
}
