import sun.reflect.generics.tree.Tree;

import java.lang.reflect.Array;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;

public class Main {
    public static int findPeakElement(int[] nums) {
            if(nums == null || nums.length == 0){
                return -1;
            }
            if(nums.length == 1 || nums[0] > nums[1]){
                return 0;
            }

            for(int i = 1 ; i < nums.length - 1; i++){
                if(nums[i] > nums[i - 1] && nums[i] > nums[i + 1]){
                    return i;
                }
            }
            return nums.length - 1;
    }

    public int peakIndexInMountainArray(int[] A) {
        if(A == null || A.length == 0){
            return -1;
        }

        if(A.length == 1 || A[0] > A[1]){
            return 0;
        }
        for(int i = 1 ; i < A.length - 1 ; i++){
            if(A[i] > A[i - 1] && A[i] > A[i + 1]){
                return i;
            }
        }
        return A.length - 1;
    }
    public static int shipWithinDays(int[] weights, int D) {
        if(weights == null || weights.length == 0){
            return 0;
        }
        int sum = 0;
        for(int weight : weights){
            sum += weight;
        }

        for(int weight = weights[0];weight <= sum ; weight++){
            int index = 0;
            for(int i = 0 ; i < D ; i++){
                int temp = 0;
                int j = index;
                for(; j < weights.length ; j++){
                    temp += weights[j];
                    if(temp <= weight){
                        continue;
                    }else{
                        break;
                    }
                }
                index = j;
            }
            if(index == weights.length){
                return  weight;
            }
        }
        return sum;
    }

    public static List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        subsetsWithDupDFS(nums,0,result,new ArrayList<>());
        return result;
    }
    public static void subsetsWithDupDFS(int[] nums,int index, List<List<Integer>> result,List<Integer> temp){
        result.add(new ArrayList<>(temp));
        boolean flag = false;
        List<Integer> list = new ArrayList<>(temp);
        for(int i = index ; i < nums.length ; i++){
            if(!flag || nums[i] != nums[i - 1]){
                list.add(nums[i]);
                subsetsWithDupDFS(nums,i + 1,result,list);
                list.remove(list.size() - 1);
                flag = true;
            }
        }
    }


    public static  int maxWidthRamp(int[] A) {
        int width = A.length - 1;
        for(;width > 0 ;width--){
            for(int i = 0 ; i + width < A.length ; i++){
                if(A[i] <= A[i + width]){
                    return width;
                }
            }
        }
        return width;
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        subsetsDFS(nums,0,result,new ArrayList<>());
        return result;
    }

    public void subsetsDFS(int[] nums,int index,List<List<Integer>> result,List<Integer> temp){
        result.add(new ArrayList<>(temp));
        List<Integer> list = new ArrayList<>(temp);
        for(int i = index ; i < nums.length ; i++){
            list.add(nums[i]);
            subsetsDFS(nums,i + 1, result, list);
            list.remove(list.size() - 1);
        }
    }
    public static List<String> letterCasePermutation(String S) {
        char[] chars = S.toCharArray();
        Set<String> result = new HashSet<>();
        letterCasePermutationDFS(chars,0,result);
        return new ArrayList<>(result);
    }
    public static void letterCasePermutationDFS(char[] chars,int index,Set<String> result){
        if(index == chars.length){
            result.add(new String(chars));
            return;
        }
        if(chars[index] >= '0' && chars[index] <= '9'){
            letterCasePermutationDFS(chars,index + 1, result);
        }else{
            chars[index] = Character.toUpperCase(chars[index]);
            letterCasePermutationDFS(chars,index + 1, result);
            chars[index] = Character.toLowerCase(chars[index]);
            letterCasePermutationDFS(chars,index + 1, result);
        }
    }

    public static List<List<Integer>> permuteUnique(int[] nums) {
        Set<List<Integer>> result = new HashSet<>();
        permuteUniqueDFS(nums,new boolean[nums.length],0,result,new ArrayList<>());
        return new ArrayList<>(result);
    }

    public static void permuteUniqueDFS(int[] nums, boolean[] used, int index,Set<List<Integer>> result, List<Integer> temp){
        if(index == nums.length){
            result.add(new ArrayList<>(temp));
            return;
        }
        List<Integer> list = new ArrayList<>(temp);
        for(int i = 0 ; i < nums.length ; i++){
            if(!used[i]){
                used[i] = true;
                list.add(nums[i]);
                permuteUniqueDFS(nums,used,index + 1,result,list);
                list.remove(list.size() - 1);
                used[i] = false;
            }

        }
    }
    public static int dominantIndex(int[] nums) {
        if(nums == null || nums.length == 0){
            return -1;
        }
        if(nums.length == 1){
            return 0;
        }
        int first = -1;
        int second = -1;

        for(int i = 0 ; i < nums.length ; i++){
            if(first == -1 || nums[first] < nums[i]){
                second = first;
                first = i;
            }else if(second == -1 || nums[second] < nums[i]){
                second = i;
            }
        }

        if(nums[second] == 0 || nums[first] / nums[second] >= 2){
            return first;
        }else{
            return -1;
        }
    }
    public  static int numFriendRequests(int[] ages) {
        int count = 0;
        int[] allAge = new int[121];
        for(int age : ages){
            allAge[age]++;
        }

        for(int i = 120 ; i >= 0 ; i--){
            if(allAge[i] > 0){

                double limit =(0.5 * i) + 7;
                if(i > limit){
                    count += allAge[i] * (allAge[i] - 1);
                }
                for(int j = i - 1 ; j > limit; j--){
                    count += allAge[i] * allAge[j];
                }
            }

        }
        return count;
    }
    public ListNode removeElements(ListNode head, int val) {
        if(head == null){
            return head;
        }
        ListNode tmp = new ListNode(-1);
        tmp.next = head;
        ListNode p = head,q = tmp;
        while (p != null){
            if(p.val == val){
                q.next = p.next;
            }else{
                q = p;
            }
            p = p.next;
        }
        return tmp.next;
    }
    public void deleteNode(ListNode node) {
        ListNode p = node;
        while(p.next.next != null){
            p.val = p.next.val;
            p = p.next;
        }
        p.val = p.next.val;
        p.next = null;
    }

    public static String shortestPalindrome(String s) {
        String result = "";
        int[][] dp = new int[s.length() + 1][s.length() + 1];
        String reverseString = new StringBuffer(s).reverse().toString();
        int max = 0;
        for(int i = 0 ; i < s.length() ; i++){
            for(int j = 0 ; j < s.length() ; j++){
                if(s.charAt(i) == reverseString.charAt(j)){
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                }
            }
            if(dp[i + 1][s.length()] == i + 1){
                max = i + 1;
            }
        }
        result = reverseString.substring(0,reverseString.length() - max) + s;
        return result;
    }
    public static String longestPalindrome(String s) {
        if(s == null || "".equals(s)){
            return "";
        }
        boolean[][] dp = new boolean[s.length()][s.length()];
        int start = 0, end = 0;
        for(int j = 0 ; j < s.length() ; j++){
            for(int i = j ; i >= 0 ; i--){
                if(i == j || ((j - i == 1 || dp[i + 1][j - 1]) && s.charAt(i) == s.charAt(j))){
                    dp[i][j] = true;
                    if(j - i > end - start){
                        start = i;
                        end = j;
                    }
                }
            }
        }
        return s.substring(start,end + 1);
    }
    public static int longestPalindromeSubseq(String s) {
        int[][] dp = new int[s.length() + 1][s.length() + 1];
        String reverseString = new StringBuffer(s).reverse().toString();
        for(int i = 0 ; i < s.length() ; i++){
            for(int j = 0 ; j < s.length() ; j++){
                if(reverseString.charAt(i) == s.charAt(j)){
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                }else{
                    dp[i + 1][j + 1] = Math.max(dp[i + 1][j],dp[i][j + 1]);
                }
            }
        }
        return dp[s.length()][s.length()];
    }

    public static int shortestSubarray(int[] A, int K) {
        int length=Integer.MAX_VALUE;
        int end = 0;
        for(int i = 0 ; i < A.length ; i++){
            int sum = 0;
            end = i;
            while(sum < K && end < A.length){
                sum += A[end++];
            }
            if(sum >= K){
                length = length > end - i ? end - i : length ;
            }
            if(end == A.length){
                for(int j = i ; j < A.length;j++){
                    if(A[j] < 0){
                        i = j;
                        break;
                    }
                }
            }
        }
        return length == Integer.MAX_VALUE ? - 1 : length;
    }
    public static  String longestCommonPrefix(String[] strs) {
        if(strs == null || strs.length == 0){
            return "";
        }
        int end = strs[0].length();
        for(int i = 1 ; i < strs.length ; i++){
            int j = 0;
            for(; j < end && j < strs[i].length() ; j++){
                if(strs[0].charAt(j) != strs[i].charAt(j)){
                    break;
                }
            }
            end = j;
            if(end == 0){
                break;
            }
        }
        return strs[0].substring(0,end);
    }
    public static int countSubstrings(String s) {
        int count = 0 ;
        boolean[][] dp = new boolean[s.length()][s.length()];
        for (int j = 0; j < s.length(); j++) {
            for(int i = j ; i >= 0 ; i--){
                if(i == j ||(j - i == 1 || dp[i + 1][j - 1]) && s.charAt(i) == s.charAt(j)){
                    count++;
                    dp[i][j] = true;
                }
            }
        }
        return count;
    }
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null && l2 == null){
            return null;
        }
        if(l1 == null)
        {
            return l2;
        }
        if(l2 == null){
            return l1;
        }
        ListNode head = new ListNode(-1);
        ListNode p = l1, q = l2,t = head;

        while(p != null && q != null){
            if(p.val <= q.val){
                t.next = p;
                p = p.next;
            }else{
                t.next = q;
                q = q.next;
            }
            t = t.next;
        }

        if(p != null){
            t.next = p;
        }
        if(q != null){
            t.next = q;
        }
        return head.next;
    }

    public static  int intersectionSizeTwo(int[][] intervals) {
        if(intervals == null || intervals.length == 0){
            return 0;
        }
        for(int i = 0 ; i < intervals.length - 1 ; i++){
            for(int j = 0 ; j < intervals.length - i - 1; j++){
                if(intervals[j][1] > intervals[j + 1][1] || (intervals[j][1] == intervals[j + 1][1] && intervals[j][0] > intervals[j + 1][0])){
                    int[] tmp = intervals[j];
                    intervals[j] = intervals[j + 1];
                    intervals[j + 1] = tmp;
                }
            }
        }
        for(int i = 0 ; i < intervals.length ; i++){
            System.out.println(intervals[i][0] + "," + intervals[i][1]);
        }
        List<Integer> list = new ArrayList<>();
       for(int i = 0 ; i < intervals.length ; i++){
            if(list.size() == 0 || list.get(list.size() - 1) < intervals[i][0]){
                list.add(intervals[i][1] - 1);
                list.add(intervals[i][1]);
            }else if(list.get(list.size() - 2) < intervals[i][0]){
                if(list.get(list.size() - 1) != intervals[i][1]){
                    list.add(intervals[i][1]);
                }else{
                    list.remove(list.size() - 1);
                    list.add(intervals[i][1] - 1);
                    list.add(intervals[i][1]);
                }
            }
       }
       return list.size();
    }
    public int minEatingSpeed(int[] piles, int H) {
        if(piles == null || piles.length == 0){
            return 0;
        }
        Arrays.sort(piles);
        int max = piles[piles.length - 1];
        int min = 1;
        int time = 0;
        int mid = (max + min) / 2;
        int speed = 0;
        while(min < max){
            if(minTime(piles,mid) <= H){
                max = mid;
                speed = mid;
            }else{
                min = mid;
            }
            mid = (max + min) / 2;
        }
        return speed;
    }

    public int minTime(int[] piles, int speed){
        int time = 0;
        for(int i = 0 ; i < piles.length ; i++){
            if(piles[i] % speed != 0){
                time += piles[i] / speed + 1;
            }else{
                time += piles[i] / speed;
            }
        }
        return time;
    }
    public int[] twoSums(int[] numbers, int target) {
        int left = 0, right = numbers.length - 1;
        while(true){
            if(numbers[left] + numbers[right] < target){
                left++;
            }else if(numbers[left] + numbers[right] > target){
                right--;
            }else{
                return new int[]{numbers[left],numbers[right]};
            }
        }
    }

    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> record = new HashMap<>();
        for(int i = 0 ; i < nums.length ; i++){
            record.put(nums[i],i);
        }
        for(int i = 0 ; i < nums.length ; i++){
            int value = target - nums[i];
            if(record.get(value) != null && record.get(value) != i){
                return new int[]{i,record.get(value)};
            }
        }
        return null;
    }

    public int subarraySum(int[] nums, int k) {
        int count = 0;
        int sum = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length ; i++){
            sum += nums[i];
            if(map.get(sum - k) != null){
                count += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum,0) + 1);
        }
        return  count;
    }

    public static int longestValidParentheses(String s) {
        Stack<Integer> stack = new Stack<>();
        int bottom = 0;
        int max = 0;
        for(int i = 0 ; i < s.length() ; i++){
            if(s.charAt(i) == '('){
                stack.push(i);
            }else if(s.charAt(i) == ')' && stack.size() == 0){
                bottom = i + 1;
            }else{
                stack.pop();
                int length = 0;
                if(stack.size() == 0 ){
                    length = i - bottom + 1;
                }else{
                    length = i - stack.peek();
                }
                max = max > length ? max : length;
            }
        }
        return max;
    }

    public int maxSubArray(int[] nums) {
        int sum = 0;
        int max = Integer.MIN_VALUE;
        for(int i = 0 ; i < nums.length ; i++){
            sum += nums[i];
            if(sum > max){
                max = sum;
            }
            sum = sum < 0 ? 0 : sum;
        }
        return max;
    }

    public int maxProfit1(int[] prices) {
        int min = Integer.MAX_VALUE;
        int value = 0;
        for(int i = 0 ; i < prices.length ; i++){
            if(prices[i] < min){
                min = prices[i];
            }else if(value < prices[i] - min){
                value = prices[i] - min;
            }
        }
        return value;
    }
    public int maxProfit2(int[] prices) {
        if(prices.length == 0){
            return 0;
        }
        int[][] dp = new int[prices.length][2];
        dp[0][0] = -prices[0];
        for(int i = 1 ; i < prices.length ; i++){
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
            dp[i][0] = Math.max(dp[i - 1][0],dp[i - 1][1] - prices[i]);
        }
        return dp[prices.length - 1][1];
    }

    public static  int maxProfit3(int[] prices) {
        int count = 2;
        int[] global = new int[count + 1];
        int[][] expenditure = new int[prices.length][count + 1];
        int[][] profit = new int[prices.length][count + 1];
        for(int i = 0 ; i <= count ; i++){
            expenditure[0][i] = -prices[0];
        }
        for(int i = 1 ; i < prices.length ; i++){
            for(int j = 1 ; j <= count ; j++){
                profit[i][j] = Math.max(global[j - 1],expenditure[i - 1][j] + prices[i]);
                global[j] = Math.max(global[j],profit[i][j]);
                expenditure[i][j] = Math.max(expenditure[i - 1][j], global[j - 1] - prices[i]);
            }
        }
        return global[count];
    }
    public int maxProfit(int k, int[] prices) {
        int count = k;
        if(k >= prices.length/2) {
            int max = 0;
            for(int i = 1; i < prices.length; ++i) {
                if(prices[i] > prices[i-1])
                    max += prices[i] - prices[i-1];
            }
            return max;
        }
        int[] global = new int[count + 1];
        int[][] expenditure = new int[prices.length][count + 1];
        int[][] profit = new int[prices.length][count + 1];
        for(int i = 0 ; i <= count ; i++){
            expenditure[0][i] = -prices[0];
        }
        for(int i = 1 ; i < prices.length ; i++){
            for(int j = 1 ; j <= count ; j++){
                profit[i][j] = Math.max(global[j - 1],expenditure[i - 1][j] + prices[i]);
                global[j] = Math.max(global[j],profit[i][j]);
                expenditure[i][j] = Math.max(expenditure[i - 1][j], global[j - 1] - prices[i]);
            }
        }
        return global[count];
    }

    public int maxProfit(int[] prices) {
        if(prices.length == 0){
            return 0;
        }
        int[] cold = new int[prices.length];
        int[] profit = new int[prices.length];
        int[] expenditure = new int[prices.length];
        expenditure[0] = -prices[0];
        for(int i = 1 ; i < prices.length ; i++){
            profit[i] = Math.max(profit[i - 1], expenditure[i - 1] + prices[i]);
            expenditure[i] = Math.max(cold[i - 1] - prices[i],expenditure[i - 1]);

            cold[i] = Math.max(profit[i - 1], Math.max(cold[i - 1] - prices[i],expenditure[i - 1]));
        }
        return profit[prices.length - 1];
    }

    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        dp[0][0] = 1;
        for(int i = 0 ; i < m ; i++){
            for(int j = 0 ; j < n ; j++){
                if(i + 1 < m){
                    dp[i + 1][j] += dp[i][j];
                }

                if(j + 1 < n){
                    dp[i][j + 1] += dp[i][j];
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length, n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = 1;
        for(int i = 0 ; i < m ; i++){
            for(int j = 0 ; j < n ; j++){
                if(obstacleGrid[i][j] != 1){
                    if(i + 1 < m){
                        dp[i + 1][j] += dp[i][j];
                    }

                    if(j + 1 < n){
                        dp[i][j + 1] += dp[i][j];
                    }
                }
            }
        }
        return dp[m - 1][n - 1];
    }
    public int minPathSum(int[][] grid) {
        int m = grid.length , n = grid[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if(i - 1 >= 0 && j - 1 >= 0){
                    grid[i][j] += Math.min(grid[i - 1][j],grid[i][j - 1]);
                }else if(i - 1 >= 0){
                    grid[i][j] += grid[i - 1][j];
                }else if(j - 1 >= 0){
                    grid[i][j] += grid[i][j - 1];
                }
            }
        }
        return grid[m - 1][n - 1];
    }

    public int calculateMinimumHP(int[][] dungeon) {
        int m = dungeon.length, n = dungeon[0].length;
        dungeon[m - 1][n - 1] = dungeon[m - 1][n - 1] > 0 ? 1 : -dungeon[m - 1][n - 1] + 1;
        for(int i = m - 1; i >= 0 ; i--){
            for(int j = n - 1 ; j >= 0 ; j--){
                if(i + 1 < m && j + 1 < n){
                    int i1 = dungeon[i + 1][j] - dungeon[i][j];
                    int j1 = dungeon[i][j + 1] - dungeon[i][j];
                    dungeon[i][j] =  Math.min(i1 > 0 ? i1 : 1,j1 > 0 ? j1 : 1);
                }else if(i + 1 < m){
                    int i1 = dungeon[i + 1][j] - dungeon[i][j];
                    dungeon[i][j] =  i1 > 0 ? i1 : 1;
                }else if(j + 1 < n){
                    int j1 = dungeon[i][j + 1] - dungeon[i][j];
                    dungeon[i][j] =  j1 > 0 ? j1 : 1;
                }
            }
        }
        return dungeon[0][0];
    }

    public int cherryPickup(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(grid[i][j] == -1){
                    continue;
                }
                if(i - 1 >= 0 && j - 1 >= 0){
                    if(grid[i - 1][j] == -1 && grid[i][j - 1] == -1){
                        grid[i][j] = -1;
                    }else if(grid[i - 1][j] == -1){
                        grid[i][j] += grid[i][j - 1];
                    }else if(grid[i][j - 1] == -1){
                        grid[i][j] += grid[i - 1][j];
                    }else{
                        grid[i][j] += Math.max(grid[i - 1][j],grid[i][j - 1]);
                    }
                }else if(i - 1 >= 0){
                    if(grid[i - 1][j] != -1){
                        grid[i][j] += grid[i][j];
                    }else{
                        grid[i][j] = -1;
                    }
                }else if(j - 1 >= 0){
                    if(grid[i][j - 1] != -1){
                        grid[i][j - 1] += grid[i][j];
                    }else{
                        grid[i][j] = -1;
                    }
                }
            }
        }
        return grid[m - 1][n - 1] == -1 ? 0 : grid[m - 1][n - 1];
    }
    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        if(n == 0 || n == 1){
            return 1;
        }
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2 ; i <= n ; i++){
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
    public int minDistance(String word1, String word2) {
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        for(int i = 1 ; i <= word2.length() ; i++){
            dp[0][i] = i;
        }
        for(int i = 1 ; i <= word1.length() ; i++){
            dp[i][0] = i;
        }

        for(int i = 1 ; i <= word1.length() ; i++){
            for(int j = 1 ; j <= word2.length() ; j++){
                if(word1.charAt(i - 1) == word2.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1];
                }else{
                    dp[i][j] = 1 + Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                }
            }
        }
        return dp[word1.length()][word2.length()];
    }
    public int minimumDeleteSum(String s1, String s2) {
        int[][] dp = new int[s1.length() + 1][s2.length() + 1];
        for(int i = 1 ; i <= s2.length() ; i++){
            dp[0][i] = (int)s2.charAt(i - 1);
        }
        for(int i = 1; i <= s1.length() ; i++){
            dp[i][0] = (int)s1.charAt(i - 1);
        }
        for(int i = 1 ; i <= s1.length() ; i++){
            for(int j = 1 ; j <= s2.length() ; j++){
                if(s1.charAt(i - 1) == s2.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1];
                }else{
                    dp[i][j] = Math.min(dp[i - 1][j] + (int)s1.charAt(i - 1), dp[i][j - 1] + (int)s2.charAt(j - 1));
                }
            }
        }
        return dp[s1.length()][s2.length()];
    }
    public static int largestRectangleArea(int[] heights) {
        int max = 0;
        Stack<Integer> stack = new Stack<>();
        for(int i = 0 ; i < heights.length ; i++){
            if(stack.isEmpty() || stack.peek() <= heights[i]){
                stack.push(heights[i]);
            }else{
                int count = 0 ;
                while(!stack.isEmpty() && heights[i] < stack.peek()){
                    count++;
                    if(max < count * stack.peek()){
                        max = count * stack.peek();
                    }
                    stack.pop();
                }

                for(int j = 0 ; j <= count ; j++){
                    stack.push(heights[i]);
                }
            }
        }
        int count = 0;
        while(!stack.isEmpty()){
            count++;
            if(max < count * stack.peek()){
                max = count * stack.peek();
            }
            stack.pop();
        }
        return max;
    }
    public static int maximalRectangle(char[][] matrix) {
        if(matrix.length == 0){
            return 0;
        }
        int max = 0;
        int[][] content = new int[matrix.length][matrix[0].length];
        for(int i = 0 ; i < matrix.length ; i++){
            if(i != 0){
                for(int j = 0 ; j < matrix[0].length ; j++){
                    if(matrix[i][j] != '0'){
                        content[i][j] = 1 + content[i - 1][j];
                    }
                }
            }else{
                for(int j = 0 ; j < matrix[0].length ; j++){
                    if(matrix[i][j] == '1'){
                        content[i][j] = 1;
                    }
                }
            }
            int tmp = largestRectangleArea(content[i]);
            max = tmp > max ? tmp : max;
        }
        return max;
    }
    public static boolean isScramble(String s1, String s2) {
        if(s1.equals(s2)){
            return true;
        }
        if(s1.length() == 0 || s2.length() == 0){
            return true;
        }
        if(s1.length() != s2.length()){
            return false;
        }
        if(s1.length() == 1){
            return  s1.equals(s2);
        }
        int mid = s1.length() / 2;
        if((isScramble(s1.substring(0,mid),s2.substring(0,mid)) && isScramble(s1.substring(mid,s1.length()),s2.substring(mid,s2.length()))) ||
                (isScramble(s1.substring(0,mid),s2.substring(s2.length() - mid,s2.length())) && isScramble(s1.substring(mid,s1.length()),s2.substring(0,s2.length() - mid)))||
                (isScramble(s1.substring(0,s1.length() - mid),s2.substring(0,s2.length() - mid)) && isScramble(s1.substring(s1.length() - mid, s1.length()),s2.substring(s2.length() - mid, s2.length())))||
                (isScramble(s1.substring(0,s1.length() - mid),s2.substring(mid,s2.length())) && isScramble(s1.substring(s1.length() - mid, s1.length()),s2.substring(0,mid)))){
            return true;
        }
        return false;
    }
    public int numJewelsInStones(String J, String S) {
        Set<Character> record = new HashSet<>();
        for(int i = 0 ; i < J.length() ; i++){
            record.add(J.charAt(i));
        }
        int count = 0;
        for(int i = 0 ; i < S.length() ; i++){
            if(record.contains(S.charAt(i))){
                count++;
            }
        }
        return count;
    }
    public Node copyRandomList(Node head) {
        if(head == null){
            return null;
        }
        Node p = head;
        while(p != null){
            Node newNode = new Node();
            newNode.val = p.val;
            newNode.next = p.next;
            p.next = newNode;
            p = newNode.next;
        }

        //复制随机指针
        p = head;
        while(p != null){
            p.next.random = p.random.next;
            p = p.next.next;
        }

        //分离
        p = head;
        Node tmp = new Node();
        Node q = tmp;
        q.next = p.next;
        q = q.next;
        while(p != null){
            p.next = q.next;
            p = q.next;
            q.next = p.next;
            q = q.next;
        }
        return tmp.next;
    }
    public static  List<String> generateParenthesis(int n) {
        Set<String> result = new HashSet<>();
        produceKH(result,"",3,3);
        return new ArrayList<>(result);
    }

    public static void produceKH(Set<String> result,String tmp, int left,int right){
        if(right == 0){
            result.add(tmp);
            return;
        }
        String str = "";
        for(int i = left; i > 0 ; i--){
            str += "(";
            produceKH(result, tmp + str, i - 1,right);
        }
        str = "";
        for(int i = right ; i > left ; i--){
            str += ")";
            produceKH(result, tmp + str, left, i - 1);
        }
    }
    public String toLowerCase(String str) {
        return str.toLowerCase();
    }
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for(int i = 0 ; i < s.length() ; i++){
            if(stack.isEmpty() || stack.peek() != s.charAt(i) - 1|| stack.peek() != s.charAt(i) - 2){
                stack.push(s.charAt(i));
            }else{
                stack.pop();
            }
        }
        return stack.isEmpty();
    }
    public int[] sortedSquares(int[] A) {
        int[] result = new int[A.length];
        int index = 0, mid = 0;
        for(int i = 0 ; i < A.length ; i++){
            if(A[i] == 0){
                mid = i;
                break;
            }else if(i - 1 >= 0 && A[i - 1] < 0 && A[i] > 0){
                mid = i;
                break;
            }else if(i == 0 && A[i] > 0){
                mid = i;
                break;
            }
            mid = i;
        }

        int left = mid - 1, right = mid;
        while(left >= 0 && right < A.length){
            if(Math.abs(A[left]) < Math.abs(A[right])){
                result[index++] = A[left] * A[left];
                left--;
            }else{
                result[index++] = A[right] * A[right];
                right++;
            }
        }
        while(left >= 0){
            result[index++] = A[left] * A[left];
            left--;
        }
        while(right < A.length){
            result[index++] = A[right] * A[right];
            right++;
        }
        return result;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode tmp = new ListNode(-1);
        tmp.next = head;

        ListNode p = tmp, q = tmp;
        while(p != null && n != 0){
            p = p.next;
            n--;
        }
        while(p.next != null){
            q = q.next;
            p = p.next;
        }

        q.next = q.next.next;
        return head;


    }
    public static int numDecodings(String s) {
        if(s.length() == 0 || s.charAt(0) == '0'){
            return 0;
        }
        int[] dp = new int[s.length() + 1];
        dp[s.length()] = 1;
        dp[s.length() - 1] = 1;
        for(int i = s.length() - 2;  i >= 0 ; i--){
            if(s.charAt(i) != '0'){
                if(Integer.parseInt(s.substring(i,i + 2)) <= 26){
                    dp[i] = dp[i + 1] + dp[i + 2];
                }else{
                    dp[i] = dp[i + 1];
                }
            }
        }
        return dp[0];
    }
    public boolean lemonadeChange(int[] bills) {
        int five = 0,ten = 0;
        for(int i = 0 ; i < bills.length ; i++){
            if(bills[i] == 5){
                five++;
            }else if(bills[i] == 10){
                if(five != 0){
                    five--;
                    ten++;
                }else{
                    return false;
                }
            }else{
                if(ten >= 1 && five >= 1){
                    ten--;
                    five--;
                }else if(five >= 3){
                    five -= 3;
                }else{
                    return false;
                }
            }
        }
        return false;
    }
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for(int i = 1; i <= n ; i++){
            for(int j = 1 ; j <= i ;j++){
                dp[i] += dp[j] * dp[i - j];
            }
        }
        return dp[n];
    }
    public int distributeCandies(int[] candies) {
        Set<Integer> kinds = new HashSet<>();
        int kind = 0;
        for(int i = 0 ; i < candies.length ; i++){
            if(!kinds.contains(candies[i])){
                kind++;
                kinds.add(candies[i]);
            }
        }
        if(kind >= candies.length / 2){
            return candies.length / 2;
        }else{
            return kind;
        }
    }

    public static int numRabbits(int[] answers) {
        Arrays.sort(answers);
        int num = 0;
        for(int i = 0 ; i < answers.length ; i++){
            int count = answers[i];
            num += count + 1;
            for(int j = 1 ; i + 1 < answers.length && j <= count ; j++){
                if(answers[i + 1] == count){
                    i++;
                }else{
                    break;
                }
            }
        }
        return num;
    }

    public int maxDepth(TreeNode root) {
        if(root == null){
            return 0;
        }

        return Math.max(maxDepth(root.left),maxDepth(root.right)) + 1;
    }
    public int minDepth(TreeNode root) {

        if(root == null){
            return 0;
        }
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        if(left == 0 || right == 0){
            return Math.max(left,right) + 1;
        }else{
            return Math.min(left,right) + 1;
        }
    }
    public static List<List<Integer>> levelOrder(TreeNode root) {
        TreeNode last = root,cur = null;
        List<List<Integer>> result = new ArrayList<>();
        Queue<TreeNode> queue = new ArrayDeque<TreeNode>();
        ((ArrayDeque<TreeNode>) queue).push(root);
        List<Integer> tmp = new ArrayList<>();
        while(!queue.isEmpty()){
            TreeNode node = ((ArrayDeque<TreeNode>) queue).pop();
            tmp.add(node.val);
            if(node.left != null){
                cur = node.left;
                ((ArrayDeque<TreeNode>) queue).push(node.left);
            }

            if(node.right != null){
                cur = node.right;
                ((ArrayDeque<TreeNode>) queue).push(node.right);
            }

            if(node == last){
                result.add(tmp);
                tmp = new ArrayList<>();
                last = cur;
            }
        }
        return result;
    }
    public boolean isInterleave(String s1, String s2, String s3) {
        boolean[][] dp = new boolean[s1.length() + 1][s2.length() + 1];
        dp[0][0] = true;
        for(int i=1;i<=s1.length();i++) {
            dp[i][0] = dp[i - 1][0] && (s1.charAt(i - 1) == s3.charAt(i - 1));
        }
        for(int i=1;i<=s2.length();i++) {
            dp[0][i]=dp[0][i-1]&&(s2.charAt(i - 1)==s3.charAt(i - 1));
        }
        for(int i=1;i<=s1.length();i++){
            for(int j=1;j<=s1.length();j++){
                dp[i][j]=(dp[i-1][j] && s1.charAt(i - 1)==s3.charAt(i+j-1) ||
                        dp[i][j-1] && s2.charAt(j - 1)==s3.charAt(i+j-1));
            }
        }
        return dp[s1.length()][s2.length()];
    }

    public int numDistinct(String s, String t) {
        int[][] dp = new int[s.length() + 1][t.length() + 1];
        for(int i = 0 ; i <=  s.length() ; i++){
            dp[i][0] = 1;
        }
        for(int i = 1 ; i <= t.length() ; i++){
            for(int j = 1 ; j <= s.length() ; j++){
                if(s.charAt(j - 1) == t.charAt(i - 1) &&
                        dp[j - 1][i - 1] != 0 &&
                        (i - 2 < 0 || s.charAt(i - 1) != s.charAt(i - 2) || s.charAt(i - 1) == s.charAt(i - 2) && dp[j][i - 1] > 1)){
                    dp[j][i] = dp[j - 1][i] + 1;
                }else{
                    dp[j][i] = dp[j - 1][i];
                }
            }
        }
        return dp[s.length()][t.length()];
    }
    public int minimumTotal(List<List<Integer>> triangle) {
        for(int i = triangle.size() - 2; i >= 0 ; i--){
            for(int j = 0 ; j < triangle.get(i).size() ; j++){
                triangle.get(i).set(j,Math.min(triangle.get(i + 1).get(j),triangle.get(i + 1).get(j + 1)) + triangle.get(i).get(j));
            }
        }
        return triangle.get(0).get(0);
    }
    public int firstMissingPositive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for(int i = 0 ; i< nums.length ; i++){
            set.add(nums[i]);
        }
        int i = 1;
        for(; ; i++){
            if(!set.contains(i)){
                break;
            }
        }
        return i;
    }

    public  static  int minSwapsCouples(int[] row) {
        int count = 0 ;
        for(int i = 0 ; i < row.length ; i += 2){
            int target = row[i] % 2 == 0 ? row[i] + 1 : row[i] - 1;
            if(row[i + 1] != target){
                for(int j = i + 2 ; j < row.length ; j++){
                    if(row[j] == target){
                        row[j] = row[i + 1];
                        row[i + 1] = target;
                        count++;
                        break;
                    }
                }
            }
        }
        return count;
    }
    public int missingNumber(int[] nums) {
        int sum = 0;
        for(int i = 0 ; i < nums.length ; i++){
            sum += nums[i];
        }
        return ((1 + nums.length) * nums.length) / 2 - sum;
    }
    public int singleNumber(int[] nums) {
        int number = 0;
        for(int i = 0 ; i < nums.length ; i++){
            number ^= nums[i];
        }
        return number;
    }

    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        if (maxChoosableInteger >= desiredTotal) return true;
        if (maxChoosableInteger * (maxChoosableInteger + 1) / 2 < desiredTotal) return false;
        if (maxChoosableInteger * (maxChoosableInteger + 1) / 2 == desiredTotal){
            return maxChoosableInteger % 2 == 0 ? false : true;
        }
        return canWin(new byte[1 << maxChoosableInteger],maxChoosableInteger,desiredTotal,0) == 1 ? true : false;
    }

    public int canWin(byte[] dp, int maxChoosableInteger, int total, int used){
        if(dp[used] != 0){
            return dp[used];
        }
        for(int i = 1; i <= maxChoosableInteger ; i++){
            if((used & (1 << i)) == 0){
                if(total <= i || canWin(dp, maxChoosableInteger, total - i, used | (1 << i)) == -1){
                    dp[used] = 1;
                    return 1;
                }
            }
        }
        dp[used] = -1;
        return -1;

    }
    public static boolean PredictTheWinner(int[] nums) {
        if(nums.length == 0){
            return false;
        }
        int[][] dp = new int[nums.length][nums.length];
        for(int i = 0 ; i < nums.length ; i++){
            dp[i][i] = nums[i];
        }

        for(int i = 1; i < nums.length ; i++){
            for(int j = 0 ; j + i < nums.length ; j++){
                dp[j][j + i] = Math.max(nums[j + i] - dp[j][j + i - 1],nums[j] - dp[j + 1][j + i]);
            }
        }
        return dp[0][nums.length - 1] > 0;
    }
    public static int minCut(String s) {
        if("".equals(s)){
            return 0;
        }
        boolean[][] dp1 = new boolean[s.length()][s.length()];
        int[][] dp2 = new int[s.length()][s.length()];
        for(int i = 0 ; i < s.length() ; i++){
            Arrays.fill(dp2[i],-1);
        }
        for(int i = 0 ; i < s.length() ; i++){
            for(int j = i ; j >= 0 ; j--){
                if(i == j || (s.charAt(i) == s.charAt(j) && (i - j == 1 || dp1[j + 1][i - 1]))){
                    dp1[j][i] = true;
                    dp2[j][i] = 0;
                }else{
                    int min = Integer.MAX_VALUE;
                    for(int p = 0 ; p <= i - j - 1 ; p++){
                        int step = dp2[j][j + p] + dp2[j + p + 1][i] + 1;
                        min = min > step ? step : min;
                    }
                    dp2[j][i] = min;
                }
            }
        }
        return dp2[0][s.length() - 1];
    }

    public static List<List<String>> partition(String s) {
        List<List<String>> result = new ArrayList<>();
        boolean[][] dp = new boolean[s.length()][s.length()];
        for(int i = 0 ; i < s.length() ; i++){
            for(int j = i ; j >= 0 ; j--){
                if(j == i || s.charAt(i) == s.charAt(j) && (i - j == 1 || dp[j + 1][i - 1])){
                    dp[j][i] = true;
                }
            }
        }
        partitionDFS(dp, result, s, new ArrayList<String>(), 0);
        return result;
    }
    public static void partitionDFS(boolean[][] dp, List<List<String>> result, String s, List<String> tmp, int index){
        if(index == s.length()){
            result.add(tmp);
        }


        for(int i = index ; i < s.length() ; i++){
            List<String> list = new ArrayList<>(tmp);
            if(dp[index][i]){
                list.add(s.substring(index,i + 1));
                partitionDFS(dp, result, s, list, i + 1);
            }
        }
    }

    public int rob1(int[] nums) {
        if(nums.length == 0){
            return 0;
        }
        int[] dp = new int[nums.length + 1];
        dp[1] = nums[0];
        for(int i = 2 ; i <= nums.length ; i++){
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        }
        return dp[nums.length];
    }
    public int rob2(int[] nums) {
        if(nums.length == 0){
            return 0;
        }
        if(nums.length == 1){
            return nums[0];
        }
        int[] dp = new int[nums.length + 1];
        dp[2] = nums[1];
        for(int i = 3 ; i <= nums.length ; i++){
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        }
        int tmp = dp[nums.length];
        Arrays.fill(dp,0);
        dp[1] = nums[0];
        for(int i = 2 ; i < nums.length ; i++){
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        }
        return Math.max(dp[nums.length - 1],tmp);
    }

    public int rob(TreeNode root) {
        if(root == null){
            return 0;
        }

        Map<TreeNode, Map<Boolean, Integer>> map = new HashMap<>();
        return Math.max(rob(map, root.left, false)+ rob(map, root.right, false),rob(map, root.left, true) + rob(map, root.right, true) + root.val);

    }

    public int rob(Map<TreeNode, Map<Boolean, Integer>> map,TreeNode root, boolean flag){

        if(root == null){
            return 0;
        }
        if(map.get(root) == null){
            map.put(root, new HashMap<>());
        }
        if(flag){
            if(map.get(root) != null || map.get(root).get(true) != null){
                return map.get(root).get(true);
            }

            int max = rob(map, root.left, false) + rob(map, root.right, false);

            map.getOrDefault(root, new HashMap<Boolean, Integer>()).put(true,max);
            return max;
        }else{
            if(map.get(root) != null || map.get(root).get(true) != null){
                map.getOrDefault(root, new HashMap<Boolean, Integer>()).put(true,rob(map, root.left, false)+ rob(map, root.right, false));
            }

            if(map.get(root) != null || map.get(root).get(false) != null)
                map.getOrDefault(root, new HashMap<Boolean, Integer>()).put(false, rob(map, root.left, true) + rob(map, root.right, true) + root.val);

            return  Math.max(map.get(root).get(true),map.get(root).get(false));
        }
    }

    public int[][] flipAndInvertImage(int[][] A) {
        for(int i = 0 ; i < A.length ; i++){
            int p = 0 , q = A[i].length - 1;
            while(p < q){
                int tmp = A[i][p];
                A[i][p] = A[i][q] == 0 ? 1 : 0;
                A[i][q] = tmp == 0 ? 1 : 0;
                p++;
                q--;
            }
        }
        return A;
    }

    public int minCostClimbingStairs(int[] cost) {
        int[] dp = new int[cost.length + 1];
        dp[1] = cost[0];
        for(int i = 2 ; i <= cost.length ; i++){
            dp[i] = Math.min(dp[i - 1], dp[i - 2]) + cost[i];
        }
        return Math.min(dp[cost.length], dp[cost.length - 1]);
    }
    public static boolean wordBreak1(String s, List<String> wordDict) {
        Set<String> words = new HashSet<>(wordDict);
        boolean[][] dp = new boolean[s.length()][s.length()];
        for(int i = 0 ; i < s.length() ; i++){
            for(int j = i ; j >= 0 ; j--){
                if(words.contains(s.substring(j, i + 1))){
                    dp[j][i] = true;
                    for(int k = 0 ; k < j ; k++){
                        if(dp[k][j - 1]){
                            dp[k][i] = true;
                        }
                    }
                }
            }
        }
        return dp[0][s.length() - 1];
    }

    public List<String> wordBreak(String s, List<String> wordDict) {
        Set<String> words = new HashSet<>(wordDict);
        boolean[][] dp = new boolean[s.length()][s.length()];
        for(int i = 0 ; i < s.length() ; i++){
            for(int j = i ; j >= 0 ; j--){
                if(words.contains(s.substring(j, i + 1))){
                    dp[j][i] = true;
                    for(int k = 0 ; k < j ; k++){
                        if(dp[k][j - 1]){
                            dp[k][i] = true;
                        }
                    }
                }
            }
        }
        return null;
    }
    public void wordBeakDFS(String s, int index, List<String> result, boolean[][] dp, String tmp){
        if(index == s.length()){
            result.add(tmp.substring(0,tmp.length() - 1));
            return ;
        }
        for(int i = index ; i < s.length() ; i++){
            if(dp[index][i]){
                wordBeakDFS(s, i + 1, result, dp, tmp + s.substring(index, i + 1) + " ");
            }
        }
    }
    public int maxProduct(int[] nums) {
        int min = 0, max = 0;
        int result = Integer.MIN_VALUE;
        for(int i = 0 ; i < nums.length ; i++){
            max = max == 0 ? nums[i] : max * nums[i];
            min = min == 0 ? nums[i] : min * nums[i];

            if(max < min){
                int tmp = max;
                max = min;
                min = tmp;
            }
            if(max > result){
                result = max;
            }
        }
        return result;
    }
    public int maxCoins(int[] nums) {
        int[] nnums = new int[nums.length + 2];
        nnums[0] = 1;
        System.arraycopy(nums,0,nnums,1, nums.length);
        nnums[nums.length + 1] = 1;
        return  maxCoinsDFS(new int[nums.length + 2][nums.length + 2] , nnums, 1, nnums.length - 2);
    }

    public int maxCoinsDFS(int[][] dp, int[] nums, int start, int end){
        if(start > end ){
            return 0;
        }
        if(dp[start][end] != 0){
            return dp[start][end];
        }

        for(int mid = start ; mid <= end ; mid++){
            int left = maxCoinsDFS(dp, nums, start, mid - 1);
            int right = maxCoinsDFS(dp, nums, mid + 1, end);
            dp[start][end] = Math.max(dp[start][end],left + right + nums[mid - 1] * nums[mid] * nums[mid + 1]);
        }
        return dp[start][end];
    }
    public int coinChange(int[] coins, int amount) {
        int[][] dp = new int[2][amount + 1];
        Arrays.sort(coins);
        for(int i = 0 ; i < coins.length ; i++){
            for(int value = coins[i]; value <= amount ; value++){
                if(dp[1][value] != value){
                    dp[0][value] = dp[0][value - coins[i]] + 1;
                    dp[1][value] = dp[1][value - coins[i]] + coins[i];
                }else if(dp[1][value - coins[i]] + coins[i] == value && dp[0][value - coins[i]] + 1 < dp[0][value]){
                    dp[0][value] = dp[0][value - coins[i]] + 1;
                }
            }
        }
        return dp[1][amount] == amount ? dp[0][amount] : -1;
    }
    public int mincostTickets(int[] days, int[] costs) {
        int[] dp = new int[366];
        int[] h = new int[366];
        for(int i = 0 ; i < days.length ; i++){
            h[days[i]] = 1;
        }

        for(int i = 1 ; i < 366 ; i++){
            if(h[i] == 0){
                dp[i] = dp[i - 1];
            }else{
                dp[i] = Math.min(dp[i - 1] + costs[0], Math.min(i - 7 > 0 ? dp[i - 7] + costs[1] : costs[1], i - 30 > 0 ? dp[i - 30] + costs[2] : costs[2]));
            }
        }
        return  dp[355];
    }
    public boolean divisorGame(int N) {
        boolean[] dp = new boolean[N + 1];
        dp[1] = false;
        for(int i = 2 ; i <= N ; i++){
            for(int k = 1; k < i ; k++){
                if(i % k == 0 && !dp[i - k]){
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[N];
    }

    public boolean canCross(int[] stones) {
        boolean[][] record = new boolean[stones.length][stones.length];
        Queue<Integer> curStone = new ArrayDeque<>();
        Queue<Integer> curStoneStep = new ArrayDeque<>();
        Map<Integer, Integer> set = new HashMap<>();
        for(int i = 0 ; i < stones.length ; i++){
            set.put(stones[i],i);
        }
        curStone.add(stones[0]);
        curStoneStep.add(0);
        while(!curStone.isEmpty()){
            int cur = ((ArrayDeque<Integer>) curStone).pop();
            int k = ((ArrayDeque<Integer>) curStoneStep).pop();
            if(k - 1 > 0 && set.containsKey(cur + k - 1) && !record[set.get(cur + k  -1)][k - 1]){
                curStone.add(cur + k - 1);
                curStoneStep.add(k - 1);

            }
            if(k > 0 && set.containsKey(cur + k) && !record[set.get(cur + k )][k]){
                curStone.add(cur + k - 1);
                curStoneStep.add(k - 1);
            }
            if( set.containsKey(cur + k + 1) && !record[set.get(cur + k + 1)][k + 1]){
                curStone.add(cur + k + 1);
                curStoneStep.add(k + 1);
            }
            if(cur + k - 1 == stones[stones.length - 1] || cur + k == stones[stones.length - 1] || cur + k + 1 == stones[stones.length - 1]){
                return true;
            }
        }
        return false;

    }
    public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
        //统计不使用大礼包的总价
        int noSpecial = 0;
        for(int i = 0;i<needs.size();i++){
            noSpecial += price.get(i) * needs.get(i);
        }
        int res = noSpecial;
        //遍历每一个大礼包
        for(List<Integer> sp : special){
            //当前大礼包超过购买数量，跳过
            if(check(sp,needs)){
                //使用当前大礼包后，还有多少剩下的
                List<Integer> newNeeds = new ArrayList<>();
                for(Integer i = 0;i<sp.size() - 1;i++){
                    newNeeds.add(needs.get(i) - sp.get(i));
                }
                //剩下的购买数量递归调用本方法，获取最低价格
                int left = shoppingOffers(price,special,newNeeds);
                //使用当前大礼包和不使用相比，选价格最低的
                res = Math.min(res,left + sp.get(sp.size() - 1));
            }
        }
        return res;
    }
    private boolean check(List<Integer> special,List<Integer> needs){
        for(int i = 0;i<needs.size();i++){
            if(special.get(i) > needs.get(i)){
                return false;
            }
        }
        return true;
    }
    public int wiggleMaxLength(int[] nums) {
        int count = 0, flag = 0;
        for(int i = 1 ; i < nums.length ; i++){
            if(nums[i] == nums[i - 1]){
                continue;
            }else if(nums[i] > nums[i - 1] && (flag == 0 || flag == -1)){
                count++;
                flag = 1;
            }else if(nums[i] < nums[i - 1] && (flag == 0 || flag == 1)){
                count++;
                flag = -1;
            }
        }
        return count + 1;
    }

    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int length = 0 , start = 0;
        for(int i = 0 ; i < s.length() ; i++){
            if(map.get(s.charAt(i)) == null){
                map.put(s.charAt(i),i);

            }else{
                int next = map.get(s.charAt(i)) + 1;
                start = start > next ? start : next;
                map.put(s.charAt(i), i);
            }
            if(i - start + 1 > length){
                length = i - start + 1;
            }
        }
        return length;
    }
    public int myAtoi(String str) {
        int flag = 1;
        long num = 0;
        int j = 0;
        while (j < str.length() && str.charAt(j) == ' ') {

            j++;
        }
        for(int i = j; i < str.length() ; i++){
            if(str.charAt(i) == '-' && i == j){
                flag = -1;
            }else if(str.charAt(i) >= '0' && str.charAt(i) <= '9'){
                num = num * 10 + str.charAt(i) - '0';
            }else{
                break;
            }
            if(num * flag < Integer.MIN_VALUE){
                return Integer.MIN_VALUE;
            }else if(num * flag > Integer.MAX_VALUE){
                return Integer.MAX_VALUE;
            }
        }
        return (int)num * flag;
    }
    public static int longestArithSeqLength(int[] A) {
        int[][] dp = new int[A.length][A.length];
        int length = 0;
        for(int i = 0 ; i < A.length ; i++){
            for(int j = i + 1 ; j < A.length ; j++){
               int len = longestArithSeqLengthDFS(dp, A[j] - A[i], j, A);
               length = len + 2 > length ? len + 2: length;
            }
        }
        return length >= 3 ? length : 0;
    }

    public static int longestArithSeqLengthDFS(int[][] dp, int step, int last, int[] A){
        if(last >= A.length){
            return 0;
        }

        for(int i = last + 1 ; i < A.length ; i++){
            if(A[i] - A[last] == step){
                if(dp[last][i] != 0){
                    return dp[last][i];
                }else{
                    dp[last][i] = longestArithSeqLengthDFS(dp, step, i, A) + 1;
                    return dp[last][i];
                }
            }
        }
        return 0;
    }
    public static boolean canPartition(int[] nums) {
        Arrays.sort(nums);
        int sum = 0;
        for(int i = 0 ; i < nums.length ; i++){
            sum += nums[i];
        }
        if(sum % 2 != 0){
            return false;
        }
        return canPartition(nums, new boolean[nums.length], 2, sum / 2, sum / 2);
    }
    public boolean canPartitionKSubsets(int[] nums, int k) {
        Arrays.sort(nums);
        int sum = 0;
        for(int i = 0 ; i < nums.length ; i++){
            sum += nums[i];
        }
        if(sum % k != 0){
            return false;
        }
        return canPartition(nums, new boolean[nums.length], k, sum / k, sum / k);
    }

    public static  boolean canPartition(int[] nums, boolean[] used,int count ,int rest, int target){
        if(count == 0){
            return true;
        }
        if(rest == 0){
            return canPartition(nums, used, count - 1, target, target);
        }
        for(int i = nums.length - 1; i >= 0; i--){
            if(!used[i] && rest - nums[i] >= 0){
                used[i] = true;
                if(canPartition(nums, used, count, rest - nums[i], target)){
                    return true;
                }
                used[i] = false;
            }else if(nums[i] > target){
                return false;
            }
        }
        return false;
    }

    public int findLength(int[] A, int[] B) {
        if(A.length == 0 || B.length == 0){
            return 0;
        }
        int[][] dp = new int[A.length + 1][B.length + 1];
        for(int i = 1; i <= A.length ; i++){
            for(int j = 1 ; j <= B.length ; j++){
                if(A[i - 1] == B[j - 1]){
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else{
                    dp[i][j] = Math.max(dp[i - 1][j - 1], Math.max(dp[i - 1][j], dp[i][j - 1]));
                }
            }
        }
        return dp[A.length][B.length];
    }

    public int minSubArrayLen(int s, int[] nums) {
        int[][] dp = new int[nums.length][nums.length];
        int length = Integer.MAX_VALUE;
        for(int i = 0 ; i < nums.length ; i++){
            for(int j = i ; j >= 0 ; j--){
                if(i == j){
                    dp[i][j] = nums[i];
                }else if(dp[j + 1][i] < s){
                    dp[j][i] = dp[j + 1][i] + nums[j];
                }
                if(dp[j][i] >= s){
                    if(i - j + 1 < length){
                        length = i - j + 1;
                    }
                    break;
                }
                if(i - j + 1 >= length){
                    break;
                }
            }
        }
        return length == Integer.MAX_VALUE ? 0 : length;
    }

    public boolean canJump(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, -1);
        dp[0] = 0;
        for(int i = 0 ; i < nums.length ; i++){
            if(dp[i] == -1){
                return false;
            }
            for(int j = i == 0 ? i + 1 : nums[dp[i]] + dp[i] + 1  ; j <= i + nums[i] ; j++){
                dp[j] = i;
            }
        }
        return true;
    }

    public static int jump(int[] nums) {
        if(nums.length == 0){
            return 0;
        }
        int[][] dp = new int[2][nums.length];
        Arrays.fill(dp[0], -1);
        dp[0][0] = 0;
        for(int i = 0 ; i < nums.length ; i++){
            for(int j = i == 0 ? i + 1 : nums[dp[0][i]] + dp[0][i] + 1  ; j < nums.length && j <= i + nums[i] ; j++){
                if(dp[0][j] != -1){
                    continue;
                }else{
                    dp[0][j] = i;
                    dp[1][j] = dp[1][i] + 1;
                }
            }
        }
        return dp[1][nums.length - 1];
    }

    public int canCompleteCircuit(int[] gas, int[] cost) {
        int[] dp = new int[cost.length];
        for(int i = 0 ; i < gas.length ; i++){
            dp[i] = gas[i] - cost[i];
        }
        for(int i = 0 ; i < dp.length ; i++){
            int sum = 0;
            boolean flag = true;
            for(int j = i; flag || j != i; j = (j + 1) % dp.length){
                flag = false;
                sum += dp[j];
                if(sum < 0){
                    break;
                }
            }
            if(sum >= 0){
                return i;
            }
        }
        return -1;
    }

    public int candy(int[] ratings) {
        if(ratings.length == 0){
            return 0;
        }
        int[] dp = new int[ratings.length];
        dp[0] = 1;
        for(int i = 1; i < ratings.length ; i++){
            if(ratings[i] > ratings[i - 1]){
                dp[i] = dp[i - 1] + 1;
            }else{
                dp[i] = 1;
            }
        }

        for(int i = ratings.length - 2; i >= 0; i--){
            if(ratings[i] > ratings[i + 1] && dp[i] <= dp[i + 1]){
                dp[i] = dp[i + 1] + 1;
            }
        }
        return dp[ratings.length - 1];
    }

    public int maxProfit(int[] prices, int fee) {
        int[] buy = new int[prices.length];
        int[] sell = new int[prices.length];
        buy[0] = sell[0] - prices[0];
        for(int i = 1 ; i < prices.length ; i++){
            sell[i] = Math.max(sell[i - 1], prices[i] + buy[i - 1] - fee);
            buy[i] = Math.max(buy[i - 1], sell[i] - prices[i]);
        }
        return sell[sell.length - 1];
    }

    public static String removeDuplicateLetters(String s) {
        int[] record = new int[26];
        boolean[] used = new boolean[26];
        for(int i = 0 ; i < s.length() ; i++){
            record[s.charAt(i) - 'a']++;
        }
        Stack<Character> stack = new Stack<>();
        for(int i = 0 ; i < s.length() ; i++){

            record[s.charAt(i) - 'a']--;
            if(used[s.charAt(i) - 'a']){
                continue;
            }
            if(stack.isEmpty() || stack.peek() < s.charAt(i)){
                stack.push(s.charAt(i));
                used[s.charAt(i) - 'a'] = true;
            }else{
                while(!stack.isEmpty() && stack.peek() > s.charAt(i) && record[stack.peek() - 'a'] > 0){
                    used[stack.pop() - 'a'] = false;
                }
                stack.push(s.charAt(i));

                used[s.charAt(i) - 'a'] = true;
            }
        }
        String str = "";
        for(char ch : stack){
            str += ch;
        }
        return str;
    }
    public List<Integer> postorderTraversal(TreeNode root) {
        Stack<TreeNode> stack1 = new Stack<>();
        Stack<TreeNode> stack2 = new Stack<>();
        if(root != null){
            stack1.push(root);
        }
        while(!stack1.isEmpty()){
            TreeNode node = stack1.pop();
            stack2.push(node);
            if(node.left != null){
                stack1.push(node.left);
            }

            if(node.right != null){
                stack1.push(node.right);
            }
        }
        List<Integer> result = new ArrayList<>();
        while(!stack2.isEmpty()){
            result.add(stack2.pop().val);
        }
        return result;
    }

    public int minDeletionSize(String[] A) {
        if(A.length == 0){
            return 0;
        }
        int count = 0;
        for(int i = 0 ; i < A[0].length() ; i++){
            for(int j = 0 ; j < A.length ; j++){
                if(j - 1 >= 0 && A[j].charAt(i) < A[j - 1].charAt(i)){
                    count++;
                    break;
                }
            }
        }
        return  count;
    }
    public int minAddToMakeValid(String S) {
        Stack<Character> stack = new Stack<>();
        for(int i = 0 ; i < S.length() ; i++){
            if(stack.isEmpty() || stack.peek() != S.charAt(i) - 1){
                stack.push(S.charAt(i));
            }else{
                stack.pop();
            }
        }
        return  stack.size();
    }
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int i = 0, j = 0;
        while(i < g.length && j < s.length){
            if(g[i] <= s[j]){
                i++;
                j++;
            }else{
                j++;
            }
        }
        return i;
    }
    public int findMinArrowShots(int[][] points) {
        boolean flag = false;
        for(int i = 0 ; i < points.length - 1 && !flag; i++){
            for(int j = 0 ; j < points.length - i - 1; j++){
                if(points[j][0] > points[j + 1][0] || (points[j][0] == points[j + 1][0] && points[j][1] < points[j + 1][1])){
                    int[] tmp = points[j];
                    points[j] = points[j + 1];
                    points[j + 1] = tmp;
                    flag = false;
                }
            }
        }
        int count = 1;
        int  end = points[0][1];
        for(int i = 1 ; i < points.length ; i++){
            if(end >= points[i][0]){
                end = Math.min(end, points[i][1]);
            }else{
                count++;
                end =  points[i][1];
            }
        }
        return count;
    }

    public static String removeKdigits(String num, int k) {
        String result = "";
        Stack<Character> stack = new Stack<>();
        for(int i = 0 ; i < num.length() ; i++){
            if(stack.isEmpty() && num.charAt(i) != '0' || stack.peek() <= num.charAt(i)){
                stack.push(num.charAt(i));
            }else if(!stack.isEmpty() && stack.peek() > num.charAt(i)){
                while(!stack.isEmpty() && stack.peek() > num.charAt(i) && k > 0){
                    stack.pop();
                    k--;
                }
                if(num.charAt(i) != '0' || num.charAt(i) == '0' && !stack.isEmpty()){
                    stack.push(num.charAt(i));
                }
            }
        }

        for(char ch : stack){
            result += ch;
        }
        return "".equals(result) ? "0" : result;
    }

    public static int monotoneIncreasingDigits(int N) {
        int length = 0;
        int val = 1;
        while(N / val != 0){
            length++;
            val *= 10;
        }
        int result = N;
        int[] num = new int[length];
        val /= 10;
        for(int i = 0; i < length ; i++){
            num[i] = N / val;
            N = N % val;
            val /= 10;
            if(i > 0 && num[i] < num[i - 1]){
                num[--i]--;
                while(i > 0){
                    if(num[i] >= num[i - 1]){
                        break;
                    }else{
                        num[--i]--;
                    }
                }
                result = 0;
                for(int j = 0 ; j < length ; j++){
                    if(j <= i){
                        result = result * 10 + num[j];
                    }else{
                        result = result * 10 + 9;
                    }
                }
                return result;
            }
        }
        return result;
    }
    public static  String strWithout3a3b(int A, int B) {
        String result = "";
        while(A != 0 && B != 0){
            if(A > B){
                result += "aab";
                A -= 2;
                B--;
            }else if(A < B){
                result += "bba";
                B -= 2;
                A--;
            }else{
                if("".equals(result) || result.charAt(result.length() - 1) == 'a'){
                    result += "b";
                    B--;
                }else{
                    result += "a";
                    A--;
                }
            }
        }
        while(A > 0){
            result += "a";
            A--;
        }
        while(B > 0){
            result += "b";
            B--;
        }

        return result;
    }
    public boolean isSubsequence(String s, String t) {
        int index = 0;
        for(int i = 0 ; i < s.length() ; i++){
            while(index < t.length() && s.charAt(i) != t.charAt(index)){
                index++;
            }

            if(index >= t.length() || s.charAt(i) != t.charAt(index)){
                return false;
            }
        }
        return true;
    }
    public static void main(String[] args) {
        System.out.println(strWithout3a3b(3,1));
    }
}

class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;
      TreeNode(int x) { val = x; }
 }
class ListNode {
    int val;
     ListNode next;
     ListNode(int x) { val = x; }
}

class Node {
    public int val;
    public Node next;
    public Node random;

    public Node() {}

    public Node(int _val,Node _next,Node _random) {
        val = _val;
        next = _next;
        random = _random;
    }
}