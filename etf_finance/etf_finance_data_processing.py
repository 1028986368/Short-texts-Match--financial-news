#!/usr/bin/env python
# -*- coding: utf-8 -*-
#=====================================================================================
#
#         FILE: news_stk_shsz_hk_equi
#        USAGE: 陆港通通道持股数量统计写入主数据目录表
#  DESCRIPTION:  
#
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
#       AUTHOR: zyc
#      COMPANY: gf.com.cn
#      VERSION: 1.0
#      CREATED: 2024-07-04
#     REVIEWER: zhangyuanchun
#     REVISION: ---
#    SRC_TABLE: t02_fxr_mkt_quot
#    TGT_TABLE: news_fxr_mkt_quot
#=====================================================================================
import sys
import os
import time
from HiveTask import HiveTask
import datetime

ht = HiveTask(sys.argv)
db_name = ht.schema_name
data_today = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
data_day_str = ht.data_day_str  #如果没有指定数据时间，默认为昨天 格式为YYYY-MM-DD
data_day_int = ht.data_day_int  #如果没有指定数据时间，默认为昨天 格式为YYYYMMDD
data_month_first_str = ht.calendar.getMonthFirst()  #月初，格式为YYYY-MM-DD
data_month_last_str = ht.calendar.getMonthLast()   # 月末,每月的最后一天

#以下函数是以数据日期作为计算日期
#ht.calendar.getMonthFirst()    每月的第一天
#ht.calendar.getMonthLast()     每月的最后一天
#ht.calendar.getWeekFirst()     每周的第一天
#ht.calendar.getWeekLast()      每周的最后一天
#ht.calendar.getYearWeek()      第几周
#ht.calendar.getYearMonth()     第几月
#ht.calendar.getYearQuarter()   第几季度
#ht.calendar.getQuarterFirst()  季度的第一天
#ht.calendar.getQuarterLast()   季度的最后一天
#ht.date_today        当前日期
#ht.date_yesterday    当前日期的昨天
#ht.oneday()          参数为加减的天数，返回数据日期加减后的日期，如数据日期减一天：ht.oneday(-1)


## 功能：在已有多源模型表t02_fxr_mkt_quot的基础上，构建主数据表news_fxr_mkt_quot外汇交易行情
## 主数据表的业务主键：secu_id, trd_dt
## 
## 处理流程：
## 1、新建主数据表、血缘源表
## 2、加工：
##   2-1、根据调研结果进行全表级加工：
##        a. ???
##        b. 设定记录来源的优先级顺序，得到中间表news_fxr_mkt_quot_all_tmp
##   
##   2-2、整合重要、源独有字段:
##        a. ???
##        b. 整合源独有字段，得到中间表news_fxr_mkt_quot_value_tmp
##
## 3、根据设定的去重规则、业务主键secu_id, trd_dt进行去重
##
## 4、插补，得到中间表news_fxr_mkt_quot_process_tmp
##  
## 5、写入主数据表、主数据血缘表
##   
## 补充：
## 1、中间表的命名规范：主数据表名+加工内容+'tmp'
## 加工内容：all、value、process


sql_comment = '1、建表'
sql = """
--1、新建:
create table if not exists news_fxr_mkt_quot(

     busi_date                                           string comment  '数据日期' ,
     rec_id                                              string comment  '记录编号' ,
     secu_id                                             string comment  '统一证券编号' ,
     trd_dt                                              string comment  '交易日期' ,
     pric_open                                           string comment  '开盘价' ,
     pric_high                                           string comment  '最高价' ,
     pric_low                                            string comment  '最低价' ,
     pric_clos                                           string comment  '收盘价' ,
     pric_buy                                            string comment  '买入价' ,
     pric_sell                                           string comment  '卖出价' ,
     remark                                              string comment  '备注' ,
     rec_upd_time                                        string comment  '记录修改时间' ,
     rec_down_time                                       string comment  '记录创建时间' ,
     last_pric_clos                                      string comment  '最新收盘价(元)' ,
     pric_pre_clos                                       string comment  '昨收盘价(元)' ,
     deal_vol                                            string comment  '成交量(手)' )

comment '外汇交易行情'
partitioned by(grp_id string comment  '并行分组标识')
stored as ORC;


-----血缘数据表
CREATE TABLE IF NOT EXISTS news_metadata_mast_tbl_consg(
     level               string      comment'粒度 1-记录 2-字段 3-原子' 
	,tbl_name_cn         string      comment'主数据表中文名'
	,rec_id              string      comment'记录编号'  --与主数据表的rec_id保持一致
	,col_name            string      comment'字段英文名'
	,col_name_cn         string      comment'字段中文名'
	,src_id              string      comment'来源标识'
	,src_rec_id          struct< secu_id: string , corp_id: string ,src_rec_id : string> comment'来源记录号'
	,src_col             string      comment'来源字段'
	,src_tbl             string      comment'来源表'
	,remark              string      comment'备注'
	,rec_upd_time        string      comment'数据更新时间'
	,rec_down_time       string      comment'数据进表时间'                                                         
)comment'主数据血缘数据表'
partitioned by (tbl_name string comment'主数据表名',busi_date string comment'数据日期', grp_id string comment '并行标识' )
stored as orc;

"""
ht.exec_sql(schema_name = db_name, sql = sql, sql_comment = sql_comment)



sql_comment = '2-5、加工、去重、插补、写入'
sql = """
set hive.merge.mapfiles = true ;
set hive.merge.mapredfiles = true;
set hive.merge.size.per.task=1073741824;
set hive.merge.smallfiles.avgsize=1073741824;
set hive.merge.orcfile.stripe.level=false;
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
set hive.exec.max.created.files=10000;
set hive.exec.max.dynamic.partitions.pernode=10000;
set hive.exec.max.dynamic.partitions=10000;
set hive.auto.convert.join=false;
set hive.support.concurrency=false;

--2、加工
--2-1、根据调研结果进行全表级加工
--2-2、设定记录来源的优先级顺序
CREATE TEMPORARY TABLE news_fxr_mkt_quot_all_tmp AS
select 
   total.*,
   t1.priority as rn,                 --记录来源的优先级顺序
   t2.priority as rn_i                --字段来源的优先级顺序
from
 t02_fxr_mkt_quot total
left join 
(
    select 
      src_id,
      priority,
      row_number()over(partition by src_id order by busi_date desc) as rk
    from 
      news_metadata_mast_tbl_list
    where 
      tbl_name = 'news_fxr_mkt_quot'
      and grp_id = '01'
      and level = '1'
)t1 on concat(total.src_id,'-',total.grp_id) = t1.src_id  --从主数据目录中获取记录来源优先级
and t1.rk = 1
left join 
(
    select 
      src_id,
      priority,
      row_number()over(partition by src_id order by busi_date desc) as rk
    from 
      news_metadata_mast_tbl_list
    where 
      tbl_name = 'news_fxr_mkt_quot'
    and grp_id = '01'
    and level = '3'
)t2 on concat(total.src_id,'-',total.grp_id) = t2.src_id       --从主数据目录中获取插补字段的来源优先级
and t2.rk = 1
;

    
--2-2、整合重要和源独有字段
CREATE TEMPORARY TABLE news_fxr_mkt_quot_value_tmp as
select
  s.secu_id,s.trd_dt,               --业务主键
  --采用有规则2插补的字段
                       
  last_pric_clos_s, pric_pre_clos_s, deal_vol_s     --采用有规则1插补的字段
FROM
  (
      select
        secu_id,trd_dt
      from news_fxr_mkt_quot_all_tmp
      group by
        secu_id,trd_dt
    ) s
  
    left join
      (
      select
        secu_id,trd_dt,
        ,
        src_id
      from news_fxr_mkt_quot_all_tmp
      where rn_i = 1
    ) a1
    on s.secu_id = a1.secu_id
    
    left join
      (
      select
        secu_id,trd_dt,
        last_pric_clos as last_pric_clos_s, pric_pre_clos as pric_pre_clos_s, deal_vol as deal_vol_s,
        ,
        src_id
      from news_fxr_mkt_quot_all_tmp
      where rn_i = 2

      ) a2
    on s.secu_id = a2.secu_id
    
  ;

--3、依据记录来源的优先级顺序对secu_id进行去重
--4、插补
CREATE TEMPORARY TABLE  news_fxr_mkt_quot_process_tmp as  
select 
  kk.*,
  last_pric_clos_s, pric_pre_clos_s, deal_vol_s,     --插补源独有字段
  ,                --插补重要字段
              --重要字段的来源
from 
  (
    SELECT 
      * 
    FROM 
      (
        select 
          t.*, 
          row_number() over(partition by secu_id,trd_dt order by rn) as rk  
        from 
          news_fxr_mkt_quot_all_tmp t) tt 
    WHERE 
      tt.rk = 1                        --依据记录来源优先级进行去重
  ) kk 
left join 
  news_fxr_mkt_quot_value_tmp b         --插补
on kk.secu_id,trd_dt = b.secu_id,trd_dt
; 


--5、写入主数据表
insert overwrite table news_fxr_mkt_quot  partition( grp_id  = '01')
select
   '"""+data_day_str+"""'                                       AS busi_date                               --数据日期
   ,MD5(concat(src_id, grp_id, rec_id))                          AS rec_id                                  --记录编号
   ,secu_id                                                      AS secu_id                                 --统一证券编号
   ,trd_dt                                                       AS trd_dt                                  --交易日期
   ,pric_open                                                    AS pric_open                               --开盘价
   ,pric_high                                                    AS pric_high                               --最高价
   ,pric_low                                                     AS pric_low                                --最低价
   ,pric_clos                                                    AS pric_clos                               --收盘价
   ,pric_buy                                                     AS pric_buy                                --买入价
   ,pric_sell                                                    AS pric_sell                               --卖出价
   ,''                                                           AS remark                                  --备注
   ,rec_upd_time                                                 AS rec_upd_time                            --记录修改时间
   ,from_unixtime(unix_timestamp(), 'yyyy-MM-dd HH:mm:ss')       AS rec_down_time                           --记录创建时间
   ,last_pric_clos_s                                             AS last_pric_clos                          --最新收盘价(元)
   ,pric_pre_clos_s                                              AS pric_pre_clos                           --昨收盘价(元)
   ,deal_vol_s                                                   AS deal_vol                                --成交量(手)

from
  news_fxr_mkt_quot_process_tmp
;

----5、写入血缘数据表
----记录级别
ALTER TABLE news_metadata_mast_tbl_consg  DROP PARTITION (tbl_name='news_fxr_mkt_quot',busi_date <= '"""+ht.oneday(-3)+"""',grp_id='01');
INSERT overwrite TABLE news_metadata_mast_tbl_consg partition(tbl_name='news_fxr_mkt_quot',busi_date='"""+data_day_str+"""',grp_id='01')
SELECT 
    '1'                           as level                     --粒度
	,'外汇交易行情'               as tbl_name_cn               --主数据表中文名
	,MD5(concat(src_id, grp_id, rec_id))  as rec_id      --记录编号
	,null                           as col_name                  --字段英文名
	,null                           as col_name_cn               --字段中文名
	,concat(src_id,'-',grp_id)      as src_id                    --来源标识
	,named_struct('secu_id',secu_id,'corp_id','','src_rec_id',src_rec_id)  as src_rec_id  --来源记录号
	,null                           as src_col                   --来源字段
	,'t02_fxr_mkt_quot'                  as src_tbl                   --来源表
	,''                             as remark                    --备注
	,rec_upd_time                                                --数据更新时间
	,from_unixtime(unix_timestamp(), 'yyyy-MM-dd HH:mm:ss') as rec_down_time  --数据进表时间
from  
  news_fxr_mkt_quot_process_tmp
;


--字段级别 --源独有字段
INSERT into TABLE news_metadata_mast_tbl_consg partition(tbl_name='news_fxr_mkt_quot',busi_date='"""+data_day_str+"""',grp_id='01')
select
 '2','外汇交易行情' ,null,'last_pric_clos','最新收盘价(元)','WD-02' ,named_struct('secu_id','','corp_id','','src_rec_id','') ,'last_pric_clos','t02_fxr_mkt_quot','' ,from_unixtime(unix_timestamp(), 'yyyy-MM-dd HH:mm:ss'),from_unixtime(unix_timestamp(), 'yyyy-MM-dd HH:mm:ss')
 union all
select
 '2','外汇交易行情' ,null,'pric_pre_clos','昨收盘价(元)','WD-02' ,named_struct('secu_id','','corp_id','','src_rec_id','') ,'pric_pre_clos','t02_fxr_mkt_quot','' ,from_unixtime(unix_timestamp(), 'yyyy-MM-dd HH:mm:ss'),from_unixtime(unix_timestamp(), 'yyyy-MM-dd HH:mm:ss')
 union all
select
 '2','外汇交易行情' ,null,'deal_vol','成交量(手)','WD-02' ,named_struct('secu_id','','corp_id','','src_rec_id','') ,'deal_vol','t02_fxr_mkt_quot','' ,from_unixtime(unix_timestamp(), 'yyyy-MM-dd HH:mm:ss'),from_unixtime(unix_timestamp(), 'yyyy-MM-dd HH:mm:ss')
 ;



"""

ht.exec_sql(schema_name = db_name, sql = sql, sql_comment = sql_comment) 
