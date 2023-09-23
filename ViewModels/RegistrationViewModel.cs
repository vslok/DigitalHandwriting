using DigitalHandwriting.Commands;
using DigitalHandwriting.Context;
using DigitalHandwriting.Helpers;
using DigitalHandwriting.Models;
using DigitalHandwriting.Services;
using DigitalHandwriting.Stores;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Windows.Input;

namespace DigitalHandwriting.ViewModels
{
    public class RegistrationViewModel : BaseViewModel
    {
        public ICommand OnCheckTextBoxKeyDownEventCommand { get; set; }
        public ICommand OnCheckTextBoxKeyUpEventCommand { get; set; }
        public ICommand OnResetRegistrationWindowButtonClickCommand { get; set; }
        public ICommand OnRegistrationFinalizeButtonClickCommand { get; set; }
        public ICommand ReturnHomeCommand { get; set; }


        private int _checkTextCurrentLetterIndex = 0;

        private int _registrationStep = 0;

        private bool _isLoginBoxEnabled = true;

        private bool _isCheckTextBoxEnabled = false;

        private bool _isFinalizeButtonVisible = false;

        private Helper _helper = new Helper();

        private string _checkTextWithUpperCase => ConstStrings.CheckText.ToUpper();

        private string _userCheckText = "";

        private string _userLogin = "";

        private DateTime _lastKeyDownTime;

        private List<List<int>> _keyPressedTimes = new List<List<int>>(3);

        private List<List<int>> _beetwenKeysTimes = new List<List<int>>(3);

        private ApplicationContext db = new ApplicationContext();

        private List<string> _registratedLogins;

        public RegistrationViewModel(
            Func<HomeViewModel> homeViewModelFactory,
            NavigationStore navigationStore)
        {
            OnCheckTextBoxKeyDownEventCommand = new RelayCommand<object>(OnCheckTextBoxKeyDownEvent);
            OnCheckTextBoxKeyUpEventCommand = new RelayCommand<object>(OnCheckTextBoxKeyUpEvent);
            OnRegistrationFinalizeButtonClickCommand = new NavigateCommand<HomeViewModel>(
                new NavigationService<HomeViewModel>(
                    navigationStore,
                    () => homeViewModelFactory()), OnRegistrationFinalizeButtonClick);
            ReturnHomeCommand = new NavigateCommand<HomeViewModel>(
                new NavigationService<HomeViewModel>(
                    navigationStore,
                    () => homeViewModelFactory()));
            OnResetRegistrationWindowButtonClickCommand = new Command(OnResetRegistrationWindowButtonClick);

            for (int i = 0; i < 3; i++)
            {
                _keyPressedTimes.Add(new List<int>());
                _beetwenKeysTimes.Add(new List<int>());
            }

            db.Database.EnsureCreated();
            db.Users.Load();
            _registratedLogins = db.Users.Local.Select(user => user.Login).ToList();
        }

        public bool IsLoginBoxEnabled
        {
            get => _isLoginBoxEnabled;
            set => SetProperty(ref _isLoginBoxEnabled, value);
        }

        public bool IsCheckTextBoxEnabled
        {
            get => _isCheckTextBoxEnabled;
            set => SetProperty(ref _isCheckTextBoxEnabled, value);
        }

        public bool IsFinalizeButtonVisible
        {
            get => _isFinalizeButtonVisible;
            set => SetProperty(ref _isFinalizeButtonVisible, value);
        }

        public int RegistrationStep
        {
            get => _registrationStep;
            set
            {
                SetProperty(ref _registrationStep, value);
                if (value == 3)
                {
                    IsLoginBoxEnabled = false;
                }
            }
        }

        public string UserLogin
        {
            get => _userLogin;
            set
            {
                SetProperty(ref _userLogin, value);
                IsCheckTextBoxEnabled = !string.IsNullOrEmpty(value) && !_registratedLogins.Contains(value); ;
            }
        }

        public string UserCheckText
        {
            get => _userCheckText;
            set 
            {
                SetProperty(ref _userCheckText, value);
                if (_registrationStep == 3 && _userCheckText.Length > 8)
                {
                    IsFinalizeButtonVisible = true;
                }
            }
        }

        private void OnRegistrationFinalizeButtonClick()
        {
            User user = new User()
            {
                Login = UserLogin,
                KeyPressedTimes = JsonSerializer.Serialize(Calculations.CalculateMedianValue(_keyPressedTimes)),
                BeetwenKeysTimes = JsonSerializer.Serialize(Calculations.CalculateMedianValue(_beetwenKeysTimes)),
                Password = EncryptionService.GetPasswordHash(UserCheckText, out string salt),
                Salt = salt
            };

            db.Users.Add(user);
            db.SaveChanges();
        }

        private void OnCheckTextBoxKeyUpEvent(object props)
        {
            var e = (KeyEventArgs)props;

            if (!_helper.CheckCurrentLetterKeyPressed(e, _checkTextCurrentLetterIndex - 1, _checkTextWithUpperCase) || _registrationStep == 3)
            {
                e.Handled = true;
                return;
            }

            var keyPressedTime = (DateTime.UtcNow - _lastKeyDownTime).Milliseconds;
            _keyPressedTimes[_registrationStep].Add(keyPressedTime);

            if (_checkTextCurrentLetterIndex == _checkTextWithUpperCase.Length)
            {
                UserCheckText = "";
                _checkTextCurrentLetterIndex = 0;
                RegistrationStep++;
            }
        }

        private void OnCheckTextBoxKeyDownEvent(object props)
        {
            var e = (KeyEventArgs)props;
            if (!_helper.CheckCurrentLetterKeyPressed(e, _checkTextCurrentLetterIndex, _checkTextWithUpperCase) || _registrationStep == 3)
            {
                e.Handled = true;
                return;
            }

            if (!string.IsNullOrEmpty(_userCheckText))
            {
                var time = (DateTime.UtcNow - _lastKeyDownTime).Milliseconds;
                _beetwenKeysTimes[_registrationStep].Add(time);
            }

            _lastKeyDownTime = DateTime.UtcNow;
            _checkTextCurrentLetterIndex++;
        }

        private void OnResetRegistrationWindowButtonClick()
        {
            RegistrationStep = 0;
            IsCheckTextBoxEnabled = false;
            IsLoginBoxEnabled = true;
            IsFinalizeButtonVisible = false;
            UserLogin = "";
            UserCheckText = "";
            _checkTextCurrentLetterIndex = 0;

            _keyPressedTimes = new List<List<int>>(3);
            _beetwenKeysTimes = new List<List<int>>(3);

            for (int i = 0; i < 3; i++)
            {
                _keyPressedTimes.Add(new List<int>());
                _beetwenKeysTimes.Add(new List<int>());
            }
        }
    }
}
