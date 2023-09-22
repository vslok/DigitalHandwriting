using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace DigitalHandwriting.Helpers
{
    public class KeyEventParameters
    {
        public object sender { get; set; }

        public KeyEventArgs e { get; set; }
    }
}
