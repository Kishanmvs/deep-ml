SELECT
    day,
    amount,

    -- Running total
    SUM(amount) OVER (
        ORDER BY day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total,

    -- Moving average of current + previous 2 rows
    AVG(amount) OVER (
        ORDER BY day
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg

FROM sales
ORDER BY day;