// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-backpropagation-is-the-adjoint-method",
        
          title: "Backpropagation is The Adjoint Method",
        
        description: "Backpropagation is often introduced as something developed within the field of machine learning. However, the story is that backpropagation really is just a special case of the adjoint method. This note is in two parts. In the first part we review the adjoint method while in the second part we describe how backpropagation is a special case of the adjoint method with a structure that result in scalable computations algorithms.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/adjoint-method/";
          
        },
      },{id: "post-associative-scan",
        
          title: "Associative Scan",
        
        description: "and its relation to state-space models",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/associative-scan/";
          
        },
      },{id: "post-structured-masked-attention",
        
          title: "Structured Masked Attention",
        
        description: "and its relation to semiseparable matrices and state-space models",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/structured-masked-attention/";
          
        },
      },{id: "post-block-tridiagonal-matrices",
        
          title: "Block Tridiagonal Matrices",
        
        description: "and their relation to semiseparable matrices and state-space models",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/block-tridiagonal/";
          
        },
      },{id: "post-continuous-matrix-factorizations",
        
          title: "Continuous Matrix Factorizations",
        
        description: "and their relation to kernel approximations",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/continuous-matrix-factorization/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-successfully-defended-my-phd-thesis-sparkles-smile",
          title: 'Successfully defended my PhD thesis :sparkles: :smile:',
          description: "",
          section: "News",},{id: "news-joined-the-section-for-scientific-computing-as-a-postdoc",
          title: 'Joined the section for scientific computing as a PostDoc',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6D%70%61%73%63@%64%74%75.%64%6B", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/mipals", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/mikkelpaltorp", "_blank");
        },
      },{
        id: 'social-orcid',
        title: 'ORCID',
        section: 'Socials',
        handler: () => {
          window.open("https://orcid.org/0000-0002-4274-2614", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=OmLUeLgAAAAJ", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
