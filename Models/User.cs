using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Collections.Generic;

namespace DigitalHandwriting.Models
{
    [PrimaryKey("Id")]
    public class User
    {
        public int Id { get; }

        [Required]
        public string Login { get; set; }

        [Required]
        public List<List<double>> HSampleValues { get; set; }

        [Required]
        public List<List<double>> UDSampleValues { get; set; }

        [Required]
        public string Password { get; set; }

        [Required]
        public string Salt { get; set; }
    }
}
