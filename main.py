# watchbill planning inspired by https://developers.google.com/optimization/scheduling/employee_scheduling
# dummy variables: https://stackoverflow.com/questions/69397752/google-or-tools-solving-an-objective-function-containing-max-function-of-multip

from ortools.sat.python import cp_model
from datetime import date, timedelta
from statistics import variance
from random import randint
from numpy import sqrt
from pandas.tseries.holiday import USFederalHolidayCalendar


class Watchbill:
    def __init__(self, start_date, end_date, days_off_between_duty, all_watchstanders):
        self.start_date = start_date
        self.end_date = end_date
        self.days_off_between_duty = days_off_between_duty  # minimum number of days off between duty days
        self.all_watchstanders = all_watchstanders
        self.num_watchstanders = len(all_watchstanders)
        self.num_days = (self.end_date-self.start_date).days + 1
        self.all_days = range(self.num_days)
        self.first_monday = (7-self.start_date.weekday()) % 7  # the first day in all_days which is a monday
        # the watchstander can't stand watch on days marked True in schedule_conflicts
        # the watchstander must stand watch on days marked True in locked_in_days
        self.schedule_conflicts = [[False for _ in range(self.num_days)] for _ in range(self.num_watchstanders)]
        self.locked_in_days = [[False for _ in range(self.num_days)] for _ in range(self.num_watchstanders)]
        self.final_schedule = [[False for _ in range(self.num_days)] for _ in range(self.num_watchstanders)]
        self.day_costs = [0 for _ in range(self.num_days)]
        self.assign_day_costs()  # build the list of day costs
        print(self.day_costs)
        # decide the maximum and minimum number of days a watchstander can stand
        self.model = cp_model.CpModel()
        self.shifts = {}
        self.min_days = self.num_days // self.num_watchstanders
        if self.num_days % self.num_watchstanders == 0:
            self.max_days = self.min_days
        else:
            self.max_days = self.min_days + 1

    def assign_day_costs(self):
        """Assign costs to each day based on how bad they are."""
        saturday_value = 7  # the badness of a day off followed by another day off
        sunday_value = 6    # the badness of a day off followed by a workday
        friday_value = 5    # the badness of a workday followed by a day off
        weekday_value = 4   # the badness of a workday followed by a workday
        # all saturdays and sundays are days off
        day_off_set = set([])
        for i in range(self.num_days + 1):
            day_of_week = (i - self.first_monday) % 7
            if day_of_week in [5, 6]:
                day_off_set.add(i)
        # all US federal holidays are also days off. We get the set from Pandas.
        cal = USFederalHolidayCalendar()
        holiday_datetimes = cal.holidays(start=str(self.start_date), end=str(self.end_date)).to_pydatetime()
        holidays = [(i.date()-self.start_date).days for i in holiday_datetimes]
        for i in holidays:
            day_off_set.add(i)
        # assign each day in num_days
        for i in range(self.num_days):
            if i in day_off_set:
                if i + 1 in day_off_set:
                    self.day_costs[i] = saturday_value
                else:
                    self.day_costs[i] = sunday_value
            else:
                if i + 1 in day_off_set:
                    self.day_costs[i] = friday_value
                else:
                    self.day_costs[i] = weekday_value

    def parse_list(self, in_list, out_list):
        """
        Convert a human-readable schedule conflict/locked in days into a matrix of True/False values
        :param in_list: A schedule conflict/locked in days, ['Name', start_date, end_date (optional)]
        :param out_list: the schedule_conflicts or locked_in_days matrix
        """
        watchstander_index = self.all_watchstanders.index(in_list[0])  # find the watchstander's index
        first_day = (in_list[1] - self.start_date).days  # convert the first day to an index
        # if the list is multiple days, there will be 3 elements. In this case, mark all days in between as True
        if len(in_list) == 3:
            last_day = (in_list[2] - self.start_date).days
            for i in range(first_day, last_day + 1):
                out_list[watchstander_index][i] = True
        # otherwise, only mark the first day as True
        else:
            out_list[watchstander_index][first_day] = True

    def parse_schedule_conflict(self, conflict):
        """
        Add a schedule conflict to the model
        :param conflict: A list of the form ['Name', start_date, end_date (optional)]
        """
        self.parse_list(conflict, self.schedule_conflicts)

    def parse_locked_days(self, locked_day):
        """
        Add a list of locked in days to the model
        :param locked_day: A list of the form ['Name', start_date, end_date (optional)]
        """
        self.parse_list(locked_day, self.locked_in_days)

    def build_model(self):
        """
        Set up the optimization model using the cp_model library from ortools
        """
        self.model = cp_model.CpModel()
        self.shifts = {}
        for n, name in enumerate(self.all_watchstanders):
            for d in self.all_days:
                # the elements of shifts are boolean variables: 1 if the day is assigned to the watchstander, else 0
                self.shifts[(n, d)] = self.model.NewBoolVar('shift_n%sd%i' % (name, d))
        # only assign one shift per day. If the day is assigned already, do not assign anyone.
        for d in self.all_days:
            if any(self.final_schedule[n][d] for n, name in enumerate(self.all_watchstanders)):
                self.model.Add(sum(self.shifts[(n, d)] for n, name in enumerate(self.all_watchstanders)) == 0)
            else:
                self.model.Add(sum(self.shifts[(n, d)] for n, name in enumerate(self.all_watchstanders)) == 1)
        for n, name in enumerate(self.all_watchstanders):
            num_days_worked = []
            for d in self.all_days:
                num_days_worked.append(self.shifts[(n, d)])
            # don't assign someone who is already assigned
            if self.is_assigned(n):
                self.model.Add(sum(num_days_worked) == 0)
            # everyone's number of days is between min_days and max_days
            else:
                self.model.Add(self.min_days <= sum(num_days_worked))
                self.model.Add(sum(num_days_worked) <= self.max_days)
        # no more than one duty day every x days
        for n, name in enumerate(self.all_watchstanders):
            for d in range(self.num_days - self.days_off_between_duty):
                self.model.Add(sum(self.shifts[(n, i)] for i in range(d, d + self.days_off_between_duty + 1)) <= 1)
        for n, name in enumerate(self.all_watchstanders):
            for d in self.all_days:
                # watchstanders can't stand watch on schedule conflicts
                if self.schedule_conflicts[n][d]:
                    self.model.Add(self.shifts[(n, d)] == 0)
                # watchstanders must stand watch on locked in days
                elif self.locked_in_days[n][d]:
                    self.model.Add(self.shifts[(n, d)] == 1)
        # assign dummy variables.
        worst_deal = self.model.NewIntVar(0, sum(self.day_costs), 'worst_deal')
        best_deal = self.model.NewIntVar(0, sum(self.day_costs), 'best_deal')
        for n, name in enumerate(self.all_watchstanders):
            if not self.is_assigned(n):
                self.model.Add(worst_deal >= sum(self.day_costs[d] * self.shifts[(n, d)] for d in self.all_days))
        for n, name in enumerate(self.all_watchstanders):
            if not self.is_assigned(n):
                self.model.Add(best_deal <= sum(self.day_costs[d] * self.shifts[(n, d)] for d in self.all_days))
        mean_deviations = []
        for n, name in enumerate(self.all_watchstanders):
            dev = self.model.NewIntVar(0, sum(self.day_costs), 'deviation_'+name)
            self.model.Add(self.num_watchstanders*(dev - sum(self.day_costs[d] * self.shifts[(n, d)] for d in self.all_days)) >= - sum(self.day_costs))
            self.model.Add(self.num_watchstanders*(dev + sum(self.day_costs[d] * self.shifts[(n, d)] for d in self.all_days)) >= sum(self.day_costs))
            mean_deviations.append(dev)
        self.model.Minimize(sum(mean_deviations))

    def solve_model(self):
        """
        Use cp_tools to solve the model.
        """
        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)
        if status == cp_model.OPTIMAL:
            return solver
        else:  # we couldn't solve the model. What happened?
            # maybe there's a day when no one can stand watch, due to schedule conflicts
            for d in self.all_days:
                if all(self.schedule_conflicts[n][d] for n, _ in enumerate(self.all_watchstanders)):
                    raise Exception('There is a schedule conflict on ' + str(self.start_date+timedelta(days=d)))
            # maybe our max and min days are too restrictive. Try again with looser limits (this iterates to zero).
            if self.min_days > 0:
                self.max_days = self.max_days + 1
                self.min_days = self.min_days - 1
                self.build_model()
                return self.solve_model()
            # something else is wrong
            else:
                self.show_solution()
                raise Exception('Unable to solve model. Check the schedule constraints.')

    def assign(self, solver, watchstander):
        """
        Assign a watchstander's schedule to the final schedule
        :param solver: ortools solver with the required schedule
        :param watchstander: index of the watchstander to assign
        """
        for d in self.all_days:
            if solver.Value(self.shifts[(watchstander, d)]) == 1:
                self.final_schedule[watchstander][d] = True

    def unassign(self, watchstander):
        """
        Set the watchstander's schedule to blank on the final schedule
        :param watchstander: index of the watchstander to unassign
        """
        for d in self.all_days:
            self.final_schedule[watchstander][d] = False

    def is_assigned(self, watchstander):
        """
        Returns true if the watchstander is assigned any days on the final schedule, false otherwise.
        :param watchstander: index of the watchstander to check
        """
        return any(self.final_schedule[watchstander][d] for d in self.all_days)

    def develop(self):
        """Find an optimal watchbill of final assignments."""
        self.build_model()
        solver = self.solve_model()
        for n, name in enumerate(self.all_watchstanders):
            self.assign(solver,n)
        self.show_solution()

    def badness_list(self):
        """Returns the badness of each watchstander on the final watchbill."""
        bl = []
        for n, name in enumerate(self.all_watchstanders):
            bl.append(sum(self.day_costs[d]*self.final_schedule[n][d] for d in self.all_days))
        return bl

    def badness_sigma(self):
        """Returns the standard deviation of badness for the final watchbill (a measure of "unfairness")"""
        return sqrt(variance(self.badness_list()))

    def show_solution(self):
        """Prints a pretty rendering of the schedule."""
        day_dict = {0: "M", 1: "T", 2: "W", 3: "R", 4: "F", 5: "S", 6: "S"}
        print(" " * (1 + max(len(str(i)) for i in self.all_watchstanders)), end="")
        for i in self.all_days:
            day_num = (self.start_date + timedelta(days=i)).day
            print((" " if day_num < 10 else "") + str(day_num) + " ", end="")
        print()
        print(" " * (1 + max(len(str(i)) for i in self.all_watchstanders)), end="")
        for i in self.all_days:
            print(" " + day_dict[(i - self.first_monday) % 7] + " ", end="")
        print()
        for n, name in enumerate(self.all_watchstanders):
            print(name, end="")
            print(" " * (1 + max(len(str(i)) for i in self.all_watchstanders) - len(str(name))), end="")
            for d in self.all_days:
                if self.final_schedule[n][d]:
                    if self.schedule_conflicts[n][d] == 1:
                        print(" ! ", end="")
                    else:
                        print(" X ", end="")
                else:
                    if self.schedule_conflicts[n][d] == 1:
                        print("---", end="")
                    else:
                        print(" . ", end="")

            print("  " + str(sum(self.day_costs[d] * self.final_schedule[n][d] for d in self.all_days)))

'''
schedule_conflict_list = [['Silver', date(2022, 2, 1), date(2022, 2, 4)],
                          ['Silver', date(2022, 2, 7), date(2022, 2, 11)],
                          ['Silver', date(2022, 2, 14), date(2022, 2, 18)],
                          ['Silver', date(2022, 2, 21), date(2022, 2, 25)],
                          ['Silver', date(2022, 2, 28)],
                          ['Kolon', date(2022, 2, 4), date(2022, 2, 7)],
                          ['Ross', date(2022, 2, 1), date(2022, 2, 4)],
                          ['Furlong', date(2022, 2, 9), date(2022, 2, 11)],
                          ['Furlong', date(2022, 2, 14), date(2022, 2, 18)],
                          ['Furlong', date(2022, 2, 21), date(2022, 2, 25)],
                          ['Furlong', date(2022, 2, 28)],
                          ['Mikalchus', date(2022, 2, 1), date(2022, 2, 3)],
                          ['Baumann', date(2022, 2, 1), date(2022, 2, 3)],
                          ['Skoric', date(2022, 2, 1), date(2022, 2, 18)]]
'''


edolist = ['Kolon', 'Arnold', 'Silver', 'Mikalchus', 'Skoric', 'Furlong', 'Baumann']
schedule_conflict_list = [['Skoric', date(2022,3,1), date(2022,3,3)],
                          ['Skoric', date(2022,3,18), date(2022,3,22)],
                          ['Skoric', date(2022,3,28)],
                          ['Silver', date(2022,3,1), date(2022,3, 6)],
                          ['Mikalchus', date(2022,3,5), date(2022,3,31)],
                          ['Furlong', date(2022,3,1), date(2022,3,4)],
                          ['Furlong', date(2022,3,7), date(2022,3,11)],
                          ['Furlong', date(2022,3,14), date(2022,3,18)],
                          ['Furlong', date(2022,3,21), date(2022,3,25)],
                          ['Furlong', date(2022,3,28), date(2022,3,31)],
                          ['Baumann', date(2022,3,1), date(2022,3,9)],
                          ['Kolon', date(2022,3,6), date(2022,3,10)]]
njy = Watchbill(date(2022, 3, 1), date(2022, 3, 31), 2, edolist)

for c in schedule_conflict_list:
    njy.parse_schedule_conflict(c)

njy.develop()

'''
edolist = range(7)
njy = Watchbill(date(2022, 2, 1), date(2022, 3, 1), 5, edolist)
njy.schedule_conflicts = [[True if randint(0, 3) == 0 else False for i in range(njy.num_days)]
                          for j in range(njy.num_watchstanders)]
njy.develop()
'''