# Contribution Guide

Welcome to the PHINIX community! Here you will find a step-by-step guide on
how to contribute to PHINIX.

## Ways to contribute

There are many ways you can contribute to PHINIX and not all of them involve
touching code. Here are some good examples of different kinds of contributions
you can make to PHINIX:

- Bug fixing and feature development in our Python codebase.
- Improving our user or developer documentation (which can be found under `/docs`).
- Reviewing code and testing existing pull requests.
- Reporting issues, making feature requests and submitting detailed bug reports.
- You may also contact us at info@ximira.org and share any feedback you may have
  the project or if you have any questions that are not covered by our docs.

We invite you to contribute and help improve PHINIX! 💚

## Your first code contribution

This section presents a step-by-step guide on how to get started as a PHINIX
contributor.

### Prerequisites

This guide assumes the following:

- You have access to a computer with a Linux distribution and know the basics
  of working with the terminal/command-line.
- You have a working installation of Git and are faimiliar with the basics of
  version control systems. If you would like to brush up on your Git skills,
  we recommend [W3 Schools' Git Tutorial](https://www.w3schools.com/git/).
- You have
  [created a GitHub account](https://docs.github.com/en/get-started/quickstart/creating-an-account-on-github)
  and are familiar with
  [the basics of how GitHub works](https://docs.github.com/en/get-started/quickstart/hello-world).

### Forking PHINIX and setting up the development environment

Go through all the sections in GitHub's official
[Fork a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo?tool=webui&platform=linux).
You should apply the steps outlined therein to fork and setup the
[ximira-org/PHINIX repository](https://github.com/ximira-org/PHINIX). Next, go
through [our guide on setting up PHINIX locally for development](docs/setup_local.md)
and set up your development environment.

### Deciding what to work on

If you already have an existing improvement in mind, we recommend you
[create a GitHub issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue)
and tell us about your ideas. We would be happy to help and point you in the
right direction. If you don't know what to work on but are excited to
make your first contribution, please reach out to us at info@ximira.org
and our team can help you find something to work on.

### Creating a pull request

After you have worked on your changes and are ready to show them off to
the world, it is time to create a pull request with your changes in them.

#### Step 1: Make sure you are on a feature branch (not `main`)

It is a common best practice to use feature branches for new changes
instead of just using the `main` branch. Run `git branch` to check
if you are on `main`. If you are on `main`, run
`git checkout feature-branch-name` to switch to a feature branch. Note
that you may replace `feature-branch-name` with any name of your choice.

#### Step 2: Update your feature branch with `git rebase`

We recommend using the `git fetch` and `git rebase` commands to update
your feature branch with the latest changes. Do not use `git pull` or
`git merge` as these commands tend to create merge commits. See
[GitHub's docs on syncing a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork)
to learn how to keep your fork in sync with the upstream repository.

Run the following commands to update your feature branch (note that you
would have to replace `feature-123` with the name of your feature branch):

``` bash
$ git checkout feature-123
Switched to branch 'feature-123'

$ git fetch upstream
remote: Counting objects: 69, done.
remote: Compressing objects: 100% (23/23), done.
remote: Total 69 (delta 49), reused 39 (delta 39), pack-reused 7
Unpacking objects: 100% (69/69), done.
From https://github.com/ximira-org/PHINIX
   69fa600..43e21f6  main     -> upstream/main

$ git rebase upstream/main

First, rewinding head to replay your work on top of it...
Applying: troubleshooting tip about provisioning
```

#### Step 3: Push your changes to your remote fork

Once your local feature branch has been updated, you are now ready to push your
changes to your remote fork:

``` bash
$ git push origin feature-123
Counting objects: 6, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (4/4), done.
Writing objects: 100% (6/6), 658 bytes | 0 bytes/s, done.
Total 6 (delta 3), reused 0 (delta 0)
remote: Resolving deltas: 100% (3/3), completed with 1 local objects.
To git@github.com:yourusername/PHINIX.git
 + 2d49e2d...bfb2433 feature-123 -> feature-123
```

If you see an error such as `failed to push some refs`, you can fix it by prefixing
the name of your branch with a `+`, like so:

``` bash
$ git push origin +feature-123
Counting objects: 6, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (4/4), done.
Writing objects: 100% (6/6), 658 bytes | 0 bytes/s, done.
Total 6 (delta 3), reused 0 (delta 0)
remote: Resolving deltas: 100% (3/3), completed with 1 local objects.
To git@github.com:yourusername/PHINIX.git
 + 2d49e2d...bfb2433 feature-123 -> feature-123 (forced update)
```

This is known as a forced update. Doing this is only recommended when you are the
only person working on a branch. If you are collaborating with different people on
the same branch, we do not recommend doing forced updates as they can lead to
complicated merge conflicts for the people you are working with.

#### Step 4: Open the pull request

To open a pull request with your changes, you can simply follow
[GitHub's official docs on creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

#### Step 5: Updating your pull request

As you collaborate with reviewers and get feedback, you will inevitably need to
update your pull request. You can simply commit your changes and push them to
your remote feature branch, like so:

``` bash
$ git push origin branch-name
Counting objects: 6, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (4/4), done.
Writing objects: 100% (6/6), 658 bytes | 0 bytes/s, done.
Total 6 (delta 3), reused 0 (delta 0)
remote: Resolving deltas: 100% (3/3), completed with 1 local objects.
To git@github.com:yourusername/PHINIX.git
 * [new branch]      branch-name -> branch-name
```

You can just keep your original pull request open and your pushed changes will
be reflected in the existing pull request. You do not need to create separate
pull requests for changes that belong to the same feature branch.

It is also important to keep your pull request up-to-date with upstream changes
so that it can be merged without any conflicts. See
[GitHub's docs on syncing a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork). You may also want to check out this excellent
article on [how to rebase a pull request](https://github.com/openedx/edx-platform/wiki/How-to-Rebase-a-Pull-Request).

Congratulations! You just made your first contribution to PHINIX!

### Accessibility

Contributors are expected to collaborate with the project team to ensure that
their contributions align with the
[Web Content Accessibility Guidelines (WCAG)](https://www.w3.org/TR/WCAG21/).
We are dedicated to creating an accessible and inclusive repository, and
contributors should actively work with the team to make necessary adjustments
and modifications that enhance the project's accessibility for users with
diverse needs. Together, we aim to foster an environment that welcomes
and accommodates all users and makers.
