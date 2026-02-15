#import "@preview/basic-resume:0.2.9": *

// Put your personal information here, replacing mine
#let name = "Mikkel Paltorp"
#let location = "Copenhagen, Denmark"
#let email = "mikkel.paltorp@gmail.com"
#let github = "github.com/mipals"
#let linkedin = "linkedin.com/in/mikkelpaltorp"
#let personal-site = "mipals.github.io"

#show: resume.with(
  author: name,
  // All the lines below are optional.
  // For example, if you want to to hide your phone number:
  // feel free to comment those lines out and they will not show.
  // location: location,
  email: email,
  github: github,
  // linkedin: linkedin,
  // phone: phone,
  personal-site: personal-site,
  accent-color: "#26428b",
  font: "New Computer Modern",
  paper: "us-letter",
  author-position: center,
  personal-info-position: center,
)

/*
* Lines that start with == are formatted into section headings
* You can use the specific formatting functions if needed
* The following formatting functions are listed below
* #edu(dates: "", degree: "", gpa: "", institution: "", location: "", consistent: false)
* #work(company: "", dates: "", location: "", title: "")
* #project(dates: "", name: "", role: "", url: "")
* certificates(name: "", issuer: "", url: "", date: "")
* #extracurriculars(activity: "", dates: "")
* There are also the following generic functions that don't apply any formatting
* #generic-two-by-two(top-left: "", top-right: "", bottom-left: "", bottom-right: "")
* #generic-one-by-two(left: "", right: "")
*/

== Work Experience

#work(
  title: "PostDoctoral Researcher (SciML for Structural Health Monitoring)",
  location: "Kgs. Lyngby, Denmark",
  company: "Technical University of Denmark (DTU Compute)",
  dates: dates-helper(start-date: "2025", end-date: "Present"),
)
- Applying transfer learning techniques to predict the structural health of offshore structures.
- Keywords: Kalman Filtering, Time Series Classification, Gaussian Processes.

#work(
  title: "PostDoctoral Researcher (Conic Optimization for Nonlinear Finite Element Analysis)",
  location: "Kgs. Lyngby / Copenhagen, Denmark",
  company: "Technical University of Denmark (DTU Compute) / OPTUM CE",
  dates: 2024,
)
- Developed a custom inhouse conic interior point solver. Worked with sparse linear algebra, Rust, C++, and Julia.
- Collaboration with industry partner (OPTUM CE) implementing our solution into their FELA software.

#work(
  title: "Ph.D in Computational Acoustics",
  location: "Kgs. Lyngby, Denmark",
  company: "Technical University of Denmark (DTU Electro)",
  dates: dates-helper(start-date: "2020", end-date: "2024"),
)
- Lead the research into applying Fast Multipole Methods and H-matrices to viscothermal acoustics. Using these methods we were able to solve problems orders of magnitudes larger than previously possible.
- Developed a Boundary Element code in Julia. The code is now used at the Technical University of Munich.
- Taught Finite & Boundary Element Methods and guided student projects in the course Numerical Acoustics.


== Education

#edu(
  institution: "Technical University of Denmark (DTU)",
  location: "Kongens Lyngby, Denmark",
  dates: dates-helper(start-date: "2014", end-date: "2020"),
  degree: "B.Sc.Eng & M.Sc.Eng in Computational Science and Engineering",

  // Uncomment the line below if you want edu formatting to be consistent with everything else
  // consistent: true
)
- Took an interest in differential equations, numerical methods, and optimization algorithms.
- For my thesis I implemented scalable (i.e. linear time and storage) algorithms for Gaussian processes.

#edu(
  institution: "McGill University",
  location: "Montreal, Canada",
  dates: "2016 Fall Semester",
  degree: "Exchange Semester",

  // Uncomment the line below if you want edu formatting to be consistent with everything else
  // consistent: true
)
- Took courses in complex analysis, non-linear and partial differential equations, and mathematical biology.


== Projects

#project(
  name: "Creator",
  // Role is optional
  role: "BoundaryIntegralEquations.jl / FMM2D.jl",
  // Dates is optional
  // dates: dates-helper(start-date: "Sep 2023", end-date: "Present"),
  // URL is also optional
  // url: [#link("https://github.com/username/BoundaryIntegralEquations.jl")[asd]],
)
- Utilizes the Flatiron Institute Fast Multipole Libraries. Developed a wrapper for the 2D library (FMM2D.jl).
- Includes SOTA techniques for large-scale viscothermal acoustics that mixes sparse linear algebra with the FMM.
- Includes automatically documentation generation using Documenter.jl and Literate.jl. 

#project(
  name: "Maintainer",
  // Role is optional
  role: "Pardiso.jl",
  // Dates is optional
  // dates: dates-helper(start-date: "May 2024", end-date: "Present"),
  // URL is also optional
  // url: [#link("https://github.com/username/BoundaryIntegralEquations.jl")[asd]],
)
- A wrapper for the sparse direct solvers MKL/Panua-Pardiso. The solvers are used throughout the Julia ecosystem.


== Skills
- *Programming Languages*: Julia, Python, C, R, Rust, MatLab
- *Technologies*: PyTorch, JAX, Git, LaTeX, Typst, CI/CD, uv, Mathematical Programming (JuMP / CVXPY)

== Conferences and Workshops

#edu(
  institution: "International Congress on Acoustics (ICA2022)",
  location: "Gyeongju, South Korea",
  dates: "Oral Presentation",
  degree: "A reduced order model including viscothermal losses",
)

#edu(
  institution: "KAIST-DTU Workshop 2022",
  location: "Daejeon, South Korea",
  dates: "Oral Presentation",
  degree: "Towards Large-scale Viscothermal Acoustics Simulations using the Boundary Element Method",
)

#edu(
  institution: "Forum Acusticum 2023",
  location: "Turin, Italy",
  dates: "Oral Presentation",
  degree: "An Open-Source Boundary Element Framework for Large-scale viscothermal acoustics",
)

// == Extracurricular Activities

// #extracurriculars(
//   activity: "Organizing committee of the Copenhagen Julia user group ",
//   dates: dates-helper(start-date: "Jan 2022", end-date: "Present"),
// )
// - Our events have around 30 participants and have included speakers from Novonesis, PumasAI, and MIT.


// #extracurriculars(
//   activity: "Science Olympiad Volunteering",
//   dates: "Sep 2023 --- Present"
// )
// - Volunteer and write tests for tournaments, including LA Regionals and SoCal State \@ Caltech

// #certificates(
//   name: "OSCP",
//   issuer: "Offensive Security",
//   // url: "",
//   date: "Oct 2024",
// )
// 